import time
import cProfile
import io
import pstats
import sys

class Logger(object):
    """
    The Logger class for capturing stdout output to a file.
    
    This class is designed to be used as a context manager. It replaces
    sys.stdout with itself when created, effectively redirecting all print
    calls and other stdout output to a log file. When done, it restores
    the original stdout stream and closes the log file.

    Attributes:
        file_name: A string representing the file name for the log file.
        terminal: Backup of the original stdout stream.
        log: File object for the log file.

    Usage:

    in run_pisco.py:
    
    # Use Logger as a context manager to capture stdout output to a log file
    with Logger(f"{ex.config.datapath_out}pisco.log") as log:

        # Depending on the configuration, perform different data processing tasks
        if (ex.config.L1C) or (ex.config.L2):
            valid_indices = flag_data(ex)
        if ex.config.L1C:
            preprocess_iasi(ex, valid_indices, data_level="l1c")
        if ex.config.L2:
            preprocess_iasi(ex, valid_indices, data_level="l2")

    # After the data processing tasks are done, move the log file to the desired location
    os.replace(f"{ex.config.datapath_out}pisco.log", f"{ex.datapath_out}pisco.log")

    """

    def __init__(self, file_name):
        """
        Initializes Logger with a file name for the log file.
        """
        self.file_name = file_name
        self.terminal = None
        self.log = None

    def __enter__(self):
        """
        Backs up the original stdout stream, opens the log file and replaces stdout with itself.
        """
        self.terminal = sys.stdout
        self.log = open(self.file_name, "w")
        sys.stdout = self
        return self

    def write(self, message):
        """
        Writes a message to both the original stdout stream and the log file.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        This flush method is needed for python 3 compatibility.
        This handles the flush command by doing nothing.
        You could add any extra behavior here if necessary.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restores the original stdout stream and closes the log file.
        """
        sys.stdout = self.terminal
        self.log.close()


class Profiler:
    """
    The Profiler class encapsulates functionality for profiling Python code.

    This class uses the cProfile module for profiling and pstats module for
    managing profiling results. It provides methods to start and stop the
    profiler and to dump profiling statistics into a file.

    Attributes:
        pr: A cProfile.Profile object used for profiling.
        start_time: A float representing the time the profiling started.

    Usage:
    in run_pisco.py:
    
    if __name__ == "__main__":
        profiler = Profiler()
        profiler.start()

        main()

        profiler.stop()
        profiler.dump_stats('/path/to/file/cProfiler_output.txt')
    """

    def __init__(self):
        """
        Initializes Profiler with a cProfile.Profile object and sets start_time to None.
        """
        self.pr = cProfile.Profile()
        self.start_time = None

    def start(self):
        """
        Starts the profiler and records the current time as the start time.
        """
        self.start_time = time.time()
        self.pr.enable()

    def stop(self):
        """
        Stops the profiler, calculates the elapsed time since the profiler was started,
        and prints the elapsed time.
        """
        self.pr.disable()
        end_time = time.time()
        print(f"Elapsed time: {end_time-self.start_time} s")

    def dump_stats(self, filepath):
        """
        Dumps profiling statistics into a file.
        
        This method uses the pstats module to manage profiling results. It sorts the 
        results by cumulative time, removes directory information from module names 
        and prints statistics into a StringIO stream. It then writes the contents 
        of the stream into a file.

        Args:
            filepath: A string representing the file path where to write the profiling statistics.
        """
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s).sort_stats('cumtime')
        ps.strip_dirs()
        ps.print_stats()
        with open(filepath, 'w+') as f:
            f.write(s.getvalue())
