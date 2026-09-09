def prepare():
    if (OUT / "contract.json").exists():
        raise ValueError("Preserve prepared experiment")
    OUT.mkdir(parents=True, exist_ok=True)
    validate_frozen_helpers(ROOT / "src")
    write(OUT / "contract.json", CONTRACT)
    source = OUT / "source"
    source.mkdir()
    for script in (Path(__file__), Path(patchnet_model.__file__)):
        shutil.copy2(script, source / script.name)
    write(
        OUT / "source.json",
        {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "scripts": {p.name: sha(p) for p in source.iterdir()},
            "frozen_helpers": validate_frozen_helpers(ROOT / "src"),
            "manifest_sha256": sha(MANIFEST),
        },
    )
    inventory = read(MANIFEST)["reviewed_inventory"]
    reports, inputs, x_parts, y_parts, partners, seed_parts = [], [], [], [], [], []
    seed_count = 0
    for record in inventory:
        if record["physical_road_group"] not in ("mandiali", "jamshoro"):
            continue
        path = ROOT / record["dzx"]
        assert sha(path) == record["sha256"]
        audit = audit_reference(path)
        assert not audit["issues"]
        assert all(
            not layer["issues"]
            and layer["amplitude_match_fraction"] == 1
            and layer["time_mapping_status"] == "verified_header"
            for layer in audit["layers"]
        )
        radar = DZTFile(path.with_suffix(".DZT"))
        data = np.asarray(radar.channel(), np.float32)
        valid = processed_boundary_mask(data)
        dt, dx = radar.header.sample_interval_ns, radar.header.distance_per_trace_m
        surface = int(np.floor(-radar.header.position_ns / dt))
        metadata = read_dzx(path)
        record_input = {
            "road": record["physical_road_group"],
            "dzt": str(path.with_suffix(".DZT")),
            "dzx": str(path),
            "dzt_sha256": sha(path.with_suffix(".DZT")),
            "dzx_sha256": sha(path),
            "dt_ns": dt,
            "dx_m": dx,
            "origin_ns": radar.header.position_ns,
            "dimensions": list(data.shape),
            "coordinate_audit": audit["layers"],
            "seeds": {},
        }
        observable = (
            native_candidates(data, valid, surface) if record_input["road"] == "mandiali" else None
        )
        for order in (2, 3):
            anchors = seed_map(
                path, metadata, order, 4 if record_input["road"] == "jamshoro" else 1
            )
            record_input["seeds"][str(order)] = anchors
            if record_input["road"] != "mandiali":
                continue
            pulse = resolve_pulse(data, valid, anchors, {}, dt, ConventionalConfig())
            tolerance = max(2, pulse.lobe_samples / 4)
            seed_bank, _ = dense._packet_correlations(
                data, valid, anchors, pulse.context_radius)
            layer = next(layer for layer in metadata.layers if layer.number + 1 == order)
            reviewed = sorted(
                (p for p in layer.picks if p.channel == 0 and p.trace not in anchors),
                key=lambda p: p.trace,
            )
            skip = max(1, int(np.ceil(len(reviewed) / CONTRACT["maximum_rows_per_file_layer"])))
            reviewed = reviewed[::skip]
            rows, columns, labels, missing = [], [], [], 0
            for point in reviewed:
                candidates = np.flatnonzero(observable[point.trace])
                target = targets(data[point.trace], candidates, point.sample, tolerance)
                positive = np.flatnonzero(target[:, 0] == 1)
                negative = np.flatnonzero(
                    (target[:, 0] == 0) & (abs(candidates - point.sample) <= 4 * pulse.lobe_samples)
                )
                if not len(positive) or not len(negative):
                    missing += 1
                    continue
                # Balanced row contribution; nearest hard negatives include same-lobe timing misses.
                positive = positive[np.argsort(abs(candidates[positive] - point.sample))[:3]]
                partner_seed = min(anchors, key=lambda r: (abs(r-point.trace), r))
                negative = _negative_indices(candidates, target, point.sample,
                    pulse.lobe_samples, seed_bank[partner_seed][point.trace, candidates])
                chosen = np.r_[positive, negative]
                rows.extend([point.trace] * len(chosen))
                columns.extend(candidates[chosen])
                labels.extend(target[chosen])
            rows, columns = np.asarray(rows), np.asarray(columns)
            x, usable = patches(data, valid, rows, columns, dt, dx)
            sr, ss = np.asarray(sorted(anchors.items())).T
            seed_x, seed_valid = patches(data, valid, sr, ss, dt, dx)
            assert np.all(seed_valid)
            mix = interval_weights(rows, sr)
            partner = np.argmax(mix, axis=1) + seed_count
            seed_count += len(seed_x)
            x_parts.append(x[usable])
            y_parts.append(np.asarray(labels)[usable])
            partners.append(partner[usable])
            seed_parts.append(seed_x)
            assert not set(rows) & set(anchors)
            reports.append(
                {
                    "file": str(path),
                    "layer": order,
                    "sampled_rows": len(reviewed),
                    "rows_without_training_pair": missing,
                    "pairs": int(usable.sum()),
                    "positive_timing": int(np.asarray(labels)[usable, 0].sum()),
                    "positive_signed_lobe": int(np.asarray(labels)[usable, 1].sum()),
                    "tolerance_samples": tolerance,
                    "pulse": pulse.metadata(),
                }
            )
            print(f"Prepared {path.stem} layer{order}: {usable.sum()} training pairs", flush=True)
        inputs.append(record_input)
    np.savez(
        OUT / "training.npz",
        x=np.concatenate(x_parts),
        y=np.concatenate(y_parts),
        partner=np.concatenate(partners),
        seeds=np.concatenate(seed_parts),
    )
    write(OUT / "inputs.json", inputs)
    write(
        OUT / "training-data.json",
        {
            "reports": reports,
            "training_sha256": sha(OUT / "training.npz"),
            "roads": ["mandiali"],
            "unknown_rows_used": False,
        },
    )
