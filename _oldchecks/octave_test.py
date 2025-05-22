import shutil, subprocess

oct = shutil.which("octave-cli") or shutil.which("octave")
print("Octave exec:", oct)

# use sum(x(:)) instead of sum(x,'all')
cmd = [
    oct,
    "--eval",
    "pkg load image; x = fspecial('gaussian',5,1); disp(sum(x(:))); quit"
]
out = subprocess.check_output(cmd)
print(out.decode())