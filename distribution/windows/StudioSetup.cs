using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: System.Reflection.AssemblyTitle("Visual Inspection Studio Setup")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]

internal static class StudioSetup
{
    private const string Product = "Visual Inspection Studio";
    private const string ProgramFile = "Visual Inspection Studio.exe";
    private const string RuntimeInstaller = "WebView2RuntimeX64.exe";
    private const string Version = "1.0.0";
    private const string RuntimeKey = @"Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}";

    [STAThread]
    private static int Main(string[] args)
    {
        bool silent = Array.IndexOf(args, "--silent") >= 0;
        bool shortcuts = Array.IndexOf(args, "--no-shortcuts") < 0;
        bool launch = Array.IndexOf(args, "--no-launch") < 0;
        try
        {
            if (!Environment.Is64BitOperatingSystem)
                throw new PlatformNotSupportedException("This release requires Windows x64.");
            bool firstInstance;
            using (var guard = new System.Threading.Mutex(true, @"Local\VisualInspectionStudioSetup", out firstInstance))
            {
                if (!firstInstance) throw new InvalidOperationException("Setup is already open.");
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                if (silent)
                {
                    string executable = Install(null, shortcuts);
                    if (launch) Launch(executable);
                    return 0;
                }
                Application.Run(CreateWindow(shortcuts, launch));
            }
            return 0;
        }
        catch (Exception error)
        {
            if (!silent) MessageBox.Show(error.Message, Product + " Setup", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }

    private static Form CreateWindow(bool shortcuts, bool launch)
    {
        var form = new Form {
            Text = Product + " Setup",
            Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath),
            StartPosition = FormStartPosition.CenterScreen,
            ClientSize = new Size(550, 278),
            FormBorderStyle = FormBorderStyle.FixedDialog,
            MaximizeBox = false,
            BackColor = Color.White,
            Font = new Font("Segoe UI", 10)
        };
        var header = new Panel { Dock = DockStyle.Top, Height = 103, BackColor = Color.FromArgb(20, 40, 62) };
        var title = new Label {
            Text = Product, Font = new Font("Segoe UI", 21, FontStyle.Bold),
            ForeColor = Color.White, AutoSize = true, Location = new Point(24, 16)
        };
        var version = new Label {
            Text = "Windows x64  ·  Version " + Version, ForeColor = Color.FromArgb(182, 215, 236),
            AutoSize = true, Location = new Point(27, 62)
        };
        header.Controls.Add(title);
        header.Controls.Add(version);
        var status = new Label {
            Text = "Install the studio and its built-in PCB1 model on this computer.",
            Location = new Point(27, 120), Size = new Size(499, 49)
        };
        var progress = new ProgressBar {
            Style = ProgressBarStyle.Marquee, Location = new Point(27, 175),
            Size = new Size(496, 10), Visible = false
        };
        var install = new Button {
            Text = "Install", Location = new Point(392, 207), Size = new Size(131, 43),
            FlatStyle = FlatStyle.Flat, BackColor = Color.FromArgb(27, 88, 207), ForeColor = Color.White
        };
        var installing = false;
        form.FormClosing += (sender, args) => { if (installing) args.Cancel = true; };
        install.Click += async (sender, args) =>
        {
            installing = true;
            install.Enabled = false;
            progress.Visible = true;
            var messages = new Progress<string>(message => status.Text = message);
            try
            {
                string executable = await Task.Run(() => Install(messages, shortcuts));
                if (launch) Launch(executable);
                installing = false;
                form.Close();
            }
            catch (Exception error)
            {
                installing = false;
                install.Enabled = true;
                progress.Visible = false;
                status.Text = "Installation stopped. Review the message and try again.";
                MessageBox.Show(form, error.Message, form.Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        };
        form.Controls.AddRange(new Control[] { header, status, progress, install });
        form.AcceptButton = install;
        return form;
    }

    private static string Install(IProgress<string> messages, bool shortcuts)
    {
        string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string root = Path.GetFullPath(Environment.GetEnvironmentVariable("VQI_INSTALL_ROOT") ?? Path.Combine(local, "VQI", "app"));
        Directory.CreateDirectory(root);
        string dest = Path.Combine(root, Version);
        string executable = Path.Combine(dest, ProgramFile);
        string marker = Path.Combine(dest, ".installed");
        if (!File.Exists(marker) || !File.Exists(executable))
        {
            if (Directory.Exists(dest))
                throw new IOException("The destination is in use or was created by another installation: " + dest);
            string stage = Path.Combine(root, ".setup-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(stage);
            try
            {
                Report(messages, "Installing the application and built-in model…");
                Extract(stage);
                if (!File.Exists(Path.Combine(stage, ProgramFile)) ||
                    !File.Exists(Path.Combine(stage, "runtime", "python.exe")) ||
                    !File.Exists(Path.Combine(stage, "starter", "model.npz")))
                    throw new InvalidDataException("The installation package is incomplete.");
                File.WriteAllText(Path.Combine(stage, ".installed"), Version);
                Directory.Move(stage, dest);
            }
            finally
            {
                if (Directory.Exists(stage)) Directory.Delete(stage, true);
            }
        }

        if (!RuntimePresent())
        {
            Report(messages, "Installing Microsoft WebView2 Runtime…");
            string runtime = Path.Combine(dest, "prerequisites", RuntimeInstaller);
            using (Process setup = Process.Start(new ProcessStartInfo(runtime, "/silent /install") {
                UseShellExecute = false, CreateNoWindow = true
            }))
            {
                setup.WaitForExit();
                if (!RuntimePresent())
                    throw new InvalidOperationException("Microsoft WebView2 setup returned exit code " +
                        setup.ExitCode + ". Restart Windows if required, then run this installer again.");
            }
        }

        if (!File.Exists(Path.Combine(Environment.SystemDirectory, "msvcp140.dll")) ||
            !File.Exists(Path.Combine(Environment.SystemDirectory, "vcruntime140_1.dll")))
        {
            Report(messages, "Installing Microsoft Visual C++ Runtime…");
            string redistributable = Path.Combine(dest, "prerequisites", "VC_redist.x64.exe");
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            using (var download = new WebClient())
                download.DownloadFile("https://aka.ms/vc14/vc_redist.x64.exe", redistributable);
            try
            {
                var start = new ProcessStartInfo(redistributable, "/install /quiet /norestart") {
                    UseShellExecute = true, Verb = "runas"
                };
                using (Process setup = Process.Start(start))
                {
                    setup.WaitForExit();
                    if (!File.Exists(Path.Combine(Environment.SystemDirectory, "msvcp140.dll")) ||
                        !File.Exists(Path.Combine(Environment.SystemDirectory, "vcruntime140_1.dll")))
                        throw new InvalidOperationException("Microsoft Visual C++ setup returned exit code " +
                            setup.ExitCode + ". Restart Windows if required, then run this installer again.");
                }
            }
            finally { File.Delete(redistributable); }
        }

        if (shortcuts)
        {
            Report(messages, "Creating your Visual Inspection Studio shortcut…");
            string shortcut = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), Product + ".lnk");
            dynamic shell = Activator.CreateInstance(Type.GetTypeFromProgID("WScript.Shell"));
            dynamic entry = shell.CreateShortcut(shortcut);
            entry.TargetPath = executable;
            entry.WorkingDirectory = dest;
            entry.Description = "Open Visual Inspection Studio";
            entry.Save();
        }
        return executable;
    }

    private static void Extract(string stage)
    {
        // The packager appends a ZIP archive and a small footer to the setup executable.
        const int footerBytes = 16;
        string targetRoot = Path.GetFullPath(stage) + Path.DirectorySeparatorChar;
        using (var input = File.OpenRead(Application.ExecutablePath))
        using (var reader = new BinaryReader(input, Encoding.ASCII, true))
        {
            if (input.Length < footerBytes) throw new InvalidDataException("The installer payload is missing.");
            input.Position = input.Length - footerBytes;
            long offset = reader.ReadInt64();
            if (Encoding.ASCII.GetString(reader.ReadBytes(8)) != "VQIZIP01" ||
                offset <= 0 || offset >= input.Length - footerBytes)
                throw new InvalidDataException("The installer payload could not be read.");
            using (var payload = new PayloadStream(input, offset, input.Length - footerBytes - offset))
            using (var archive = new ZipArchive(payload, ZipArchiveMode.Read))
            {
                foreach (ZipArchiveEntry entry in archive.Entries)
                {
                    string target = Path.GetFullPath(Path.Combine(targetRoot, entry.FullName));
                    if (!target.StartsWith(targetRoot, StringComparison.OrdinalIgnoreCase))
                        throw new InvalidDataException("The installer contains an invalid path.");
                    if (entry.Name.Length == 0) Directory.CreateDirectory(target);
                    else
                    {
                        Directory.CreateDirectory(Path.GetDirectoryName(target));
                        entry.ExtractToFile(target);
                    }
                }
            }
        }
    }

    private static bool RuntimePresent()
    {
        return RegistryVersion(RegistryHive.LocalMachine, RegistryView.Registry32) ||
            RegistryVersion(RegistryHive.CurrentUser, RegistryView.Registry64) ||
            RegistryVersion(RegistryHive.CurrentUser, RegistryView.Registry32);
    }

    private static bool RegistryVersion(RegistryHive hive, RegistryView view)
    {
        using (var parent = RegistryKey.OpenBaseKey(hive, view))
        using (var key = parent.OpenSubKey(RuntimeKey))
        {
            System.Version version;
            return key != null && System.Version.TryParse(key.GetValue("pv") as string, out version) &&
                version.Major > 0;
        }
    }

    private static void Report(IProgress<string> messages, string value)
    {
        if (messages != null) messages.Report(value);
    }

    private static void Launch(string exe)
    {
        Process.Start(new ProcessStartInfo(exe) { WorkingDirectory = Path.GetDirectoryName(exe), UseShellExecute = true });
    }

    private sealed class PayloadStream : Stream
    {
        private readonly Stream source;
        private readonly long start;
        private readonly long length;
        private long position;

        public PayloadStream(Stream source, long start, long length)
        {
            this.source = source;
            this.start = start;
            this.length = length;
        }

        public override bool CanRead { get { return true; } }
        public override bool CanSeek { get { return true; } }
        public override bool CanWrite { get { return false; } }
        public override long Length { get { return length; } }
        public override long Position
        {
            get { return position; }
            set { Seek(value, SeekOrigin.Begin); }
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            if (position >= length) return 0;
            source.Position = start + position;
            int size = source.Read(buffer, offset, (int)Math.Min(count, length - position));
            position += size;
            return size;
        }
        public override long Seek(long offset, SeekOrigin origin)
        {
            long anchor = origin == SeekOrigin.Begin ? 0 : origin == SeekOrigin.Current ? position : length;
            if (anchor + offset < 0) throw new IOException("Invalid payload position.");
            return position = anchor + offset;
        }
        public override void Flush() { }
        public override void SetLength(long value) { throw new NotSupportedException(); }
        public override void Write(byte[] buffer, int offset, int count) { throw new NotSupportedException(); }
    }
}
