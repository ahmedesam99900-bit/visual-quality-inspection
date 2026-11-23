using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

[assembly: System.Reflection.AssemblyTitle("Visual Inspection Studio")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]

internal sealed class StudioWindow : Form
{
    private readonly string appDirectory = AppDomain.CurrentDomain.BaseDirectory;
    private readonly string userDirectory;
    private readonly Label startup;
    private Process backend;
    private StreamWriter log;
    private WebView2 view;
    private bool closing;

    private StudioWindow()
    {
        string root = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        userDirectory = Environment.GetEnvironmentVariable("VQI_DATA_HOME") ??
            Path.Combine(root, "VisualInspectionStudio");
        Text = "Visual Inspection Studio";
        StartPosition = FormStartPosition.CenterScreen;
        MinimumSize = new Size(680, 500);
        Size = new Size(1360, 900);
        Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
        BackColor = Color.FromArgb(243, 245, 248);

        var menu = new MenuStrip();
        var help = new ToolStripMenuItem("Help");
        help.DropDownItems.Add("Connect Modal GPU…", null, ConnectModal);
        help.DropDownItems.Add("Open diagnostic log", null, OpenLog);
        menu.Items.Add(help);
        MainMenuStrip = menu;

        startup = new Label();
        startup.Dock = DockStyle.Fill;
        startup.TextAlign = ContentAlignment.MiddleCenter;
        startup.ForeColor = Color.FromArgb(26, 48, 75);
        startup.Font = new Font("Segoe UI", 20, FontStyle.Bold);
        startup.Text = "Starting Visual Inspection Studio…";
        Controls.Add(startup);
        Controls.Add(menu);
        Shown += StartStudio;
    }

    private async void StartStudio(object sender, EventArgs args)
    {
        try
        {
            Directory.CreateDirectory(userDirectory);
            log = new StreamWriter(Path.Combine(userDirectory, "desktop.log"), true);
            log.AutoFlush = true;
            Uri address = await StartBackend();
            if (closing) return;

            startup.Text = "Opening your workspace…";
            view = new WebView2 { Dock = DockStyle.Fill };
            var environment = await CoreWebView2Environment.CreateAsync(null, Path.Combine(userDirectory, "webview"));
            if (closing) return;
            Controls.Add(view);
            view.BringToFront();
            // Leave the native menu above the embedded content.
            MainMenuStrip.BringToFront();
            await view.EnsureCoreWebView2Async(environment);
            if (closing) return;
            view.CoreWebView2.Settings.AreDevToolsEnabled = false;
            view.CoreWebView2.Settings.IsStatusBarEnabled = false;
            view.CoreWebView2.NavigationStarting += (object source, CoreWebView2NavigationStartingEventArgs e) =>
            {
                Uri target;
                e.Cancel = !Uri.TryCreate(e.Uri, UriKind.Absolute, out target) ||
                    target.Scheme != address.Scheme || target.Host != address.Host || target.Port != address.Port;
            };
            view.CoreWebView2.NewWindowRequested += (object source, CoreWebView2NewWindowRequestedEventArgs e) =>
            {
                e.Handled = true;
                Uri target;
                if (Uri.TryCreate(e.Uri, UriKind.Absolute, out target) && target.Scheme == "https")
                    Process.Start(new ProcessStartInfo(target.AbsoluteUri) { UseShellExecute = true });
            };
            view.CoreWebView2.Navigate(address.AbsoluteUri);
        }
        catch (Exception error)
        {
            if (closing) return;
            WriteLog(error.ToString());
            MessageBox.Show(this, "The studio could not start.\n\n" + error.Message +
                "\n\nLog: " + Path.Combine(userDirectory, "desktop.log"), Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
            Close();
        }
    }

    private async Task<Uri> StartBackend()
    {
        var address = new TaskCompletionSource<Uri>();
        string workspace = Path.Combine(userDirectory, "workspace");
        string script = Path.Combine(appDirectory, "deployment", "app.py");
        string model = Path.Combine(appDirectory, "starter");
        var start = new ProcessStartInfo(Path.Combine(appDirectory, "runtime", "python.exe"));
        start.Arguments = "-B -E -s -X utf8 " + Quote(script) + " --workspace " +
            Quote(workspace) + " --starter " + Quote(model) + " --port 0";
        start.WorkingDirectory = appDirectory;
        start.CreateNoWindow = true;
        start.UseShellExecute = false;
        start.RedirectStandardError = true;
        start.RedirectStandardOutput = true;
        backend = new Process { StartInfo = start, EnableRaisingEvents = true };
        backend.OutputDataReceived += (object source, DataReceivedEventArgs e) =>
        {
            if (e.Data == null) return;
            WriteLog(e.Data);
            const string prefix = "Visual Inspection is running at ";
            Uri local;
            if (e.Data.StartsWith(prefix, StringComparison.Ordinal) &&
                Uri.TryCreate(e.Data.Substring(prefix.Length), UriKind.Absolute, out local) &&
                local.Scheme == "http" && local.Host == "127.0.0.1" && local.Port > 0)
            {
                address.TrySetResult(local);
            }
        };
        backend.ErrorDataReceived += (object source, DataReceivedEventArgs e) => WriteLog(e.Data);
        backend.Exited += (object source, EventArgs e) =>
        {
            if (!address.TrySetException(new InvalidOperationException("The bundled inference service stopped during startup.")) &&
                !closing && IsHandleCreated)
            {
                try
                {
                    BeginInvoke((Action)(() => {
                        if (closing) return;
                        MessageBox.Show(this, "The inference service stopped. Reopen the studio to restart it.", Text,
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
                        Close();
                    }));
                }
                catch (InvalidOperationException) { }
            }
        };
        backend.Start();
        backend.BeginErrorReadLine();
        backend.BeginOutputReadLine();
        Task result = await Task.WhenAny(address.Task, Task.Delay(TimeSpan.FromSeconds(60)));
        if (result != address.Task) throw new TimeoutException("The bundled inference service did not become ready.");
        return await address.Task;
    }

    private void WriteLog(string message)
    {
        lock (this)
        {
            if (log != null && message != null)
            {
                try { log.WriteLine(message); }
                catch (IOException) { }
            }
        }
    }

    private void ConnectModal(object sender, EventArgs args)
    {
        string connector = Path.Combine(appDirectory, "Connect-Modal.cmd");
        Process.Start(new ProcessStartInfo(connector) { UseShellExecute = true, WorkingDirectory = appDirectory });
    }

    private void OpenLog(object sender, EventArgs args)
    {
        string path = Path.Combine(userDirectory, "desktop.log");
        if (File.Exists(path)) Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        closing = true;
        if (backend != null)
        {
            try { if (!backend.HasExited) backend.Kill(); }
            catch (InvalidOperationException) { }
        }
        lock (this) { if (log != null) { log.Dispose(); log = null; } }
        base.OnFormClosing(e);
    }

    private static string Quote(string value) { return "\"" + value + "\""; }

    [STAThread]
    private static void Main()
    {
        bool firstInstance;
        using (var instance = new System.Threading.Mutex(true, @"Local\VisualInspectionStudio", out firstInstance))
        {
            if (!firstInstance)
            {
                MessageBox.Show("Visual Inspection Studio is already open.", "Visual Inspection Studio");
                return;
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new StudioWindow());
        }
    }
}
