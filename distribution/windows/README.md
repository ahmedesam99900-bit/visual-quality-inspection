# Visual Inspection Studio for Windows

For users, distribute `outputs/windows-desktop/release/Visual-Inspection-Studio-Setup.exe`. Open it and click **Install**. Setup opens the desktop application and adds it to the Windows Start menu.

The desktop window uses Windows Forms and Microsoft WebView2. It starts the bundled Python inference process on a temporary loopback port and renders the existing interface inside the window. Closing the window stops the inference process. User workspaces and activated models stay in `%LOCALAPPDATA%\VisualInspectionStudio\workspace`.

The Windows x64 build contains the Python environment, CPU training runtime, WebView2 standalone installer and PCB1 crop starter. Setup installs WebView2 if required. If the computer lacks the Microsoft Visual C++ runtime, Setup downloads the official installer and Windows can request administrator approval.

To rebuild from this project, run the following command in PowerShell at the project root.

```powershell
.\.venv\Scripts\python.exe -B .\distribution\windows\build_installer.py
```

The build inputs are in `outputs/windows-desktop/build-cache`. Keep the tested `base-runtime.zip`, the official WebView2 SDK NuGet package `webview2.nupkg` and the official x64 WebView2 Evergreen standalone installer `WebView2RuntimeX64.exe` in that directory. The builder reads the latest application code from `deployment` and places the installer in `outputs/windows-desktop/release`.
