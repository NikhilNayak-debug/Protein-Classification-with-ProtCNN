from tensorboard import program
import webbrowser

def launch_tensorboard(logdir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir])
    url = tb.launch()
    print(f"TensorBoard is running at {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    logdir = "./lightning_logs/"  # Change this to your log directory
    launch_tensorboard(logdir)