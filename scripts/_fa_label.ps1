param(
    [Parameter(Mandatory = $true)][string]$Text,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Color = "A33B24",
    [double]$Size = 64
)
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
if (-not ("FaLabel" -as [type])) {
    Add-Type -ReferencedAssemblies System.Drawing, System.Windows.Forms -TypeDefinition @"
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
public static class FaLabel {
  public static void Save(string text, string path, int size, int r, int g, int b) {
    using (var font = new Font("B Nazanin", size, FontStyle.Bold, GraphicsUnit.Pixel)) {
      var flags = TextFormatFlags.RightToLeft | TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPadding;
      Size measured;
      using (var probe = new Bitmap(4, 4))
      using (var pg = Graphics.FromImage(probe))
        measured = TextRenderer.MeasureText(pg, text, font, new Size(int.MaxValue, int.MaxValue), flags);
      int w = Math.Max(8, measured.Width + 12);
      int h = Math.Max(8, measured.Height + 8);
      using (var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb))
      using (var gr = Graphics.FromImage(bmp)) {
        gr.Clear(Color.Transparent);
        gr.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAliasGridFit;
        TextRenderer.DrawText(gr, text, font, new Rectangle(0, 0, w, h), Color.FromArgb(r, g, b), Color.Transparent, flags);
        bmp.Save(path, ImageFormat.Png);
      }
    }
  }
}
"@
}
$r = [Convert]::ToInt32($Color.Substring(0, 2), 16)
$gch = [Convert]::ToInt32($Color.Substring(2, 2), 16)
$b = [Convert]::ToInt32($Color.Substring(4, 2), 16)
[FaLabel]::Save($Text, $Out, [int]$Size, $r, $gch, $b)
Write-Output "ok"
