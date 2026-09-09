"""One physical-distance objective shared by selection and competing-path margins."""


def local_score(segment, scores, step=1.0, transition_cost=1.0):
    return step * (
        sum(float(scores[row, col]) for row, col in segment.nodes)
        - transition_cost * segment.transition_cost
    )


def transition_penalty(before, after, quality, step=1.0, transition_cost=1.0, gap_cost=0.1):
    distance = max(1, after.start - before.stop) * step
    gap = max(0, after.start - before.stop - 1) * step
    return transition_cost * (1 - quality) * distance + gap_cost * gap
