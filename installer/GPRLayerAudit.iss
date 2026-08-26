#define AppName "GPR Layer Audit"
#define AppVersion "0.1.0"
#define AppPublisher "GPR Layer Audit Team"
#define AppExeName "GPRLayerAudit.exe"

[Setup]
AppId={{D6313767-4568-462C-826B-C084E88E84C9}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\GPR Layer Audit
DefaultGroupName={#AppName}
OutputDir=..\installer-output
OutputBaseFilename=GPR-Layer-Audit-{#AppVersion}-x64-Setup
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\{#AppExeName}
; For a signed release, configure an Inno Setup SignTool named "release" and
; uncomment: SignTool=release

[Files]
Source: "..\dist\GPRLayerAudit\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
