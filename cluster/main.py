import tkinter as tk
from tkinter import ttk
from threading import Thread, Event
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SimpyGUI(tk.Frame):
    def __init__(self, master):
        self.master = master
        tk.Frame.__init__(self, master)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.stop_event = Event()
        self.cluster_thread = None
        self.time_history = []
        self.master.title("Cluster")

         # Controls frame
        self.controls_frame = tk.Frame(master)
        self.controls_frame.pack(side='left', fill='y', padx=10, pady=10)

        # Entry field for number of samples
        self.num_samples_label = tk.Label(self.controls_frame, text="Number of samples")
        self.num_samples_label.pack()

        self.num_samples_entry = tk.Entry(self.controls_frame)
        self.num_samples_entry.insert(0, "1000")
        self.num_samples_entry.pack()

        # Checkbutton for random state
        self.random_state_var = tk.IntVar()
        self.random_state_var.set(42)
        self.random_state_checkbutton = tk.Checkbutton(self.controls_frame, text="Random state", variable=self.random_state_var)
        self.random_state_checkbutton.pack()

        # Entry field for number of clusters
        self.num_clusters_label = tk.Label(self.controls_frame, text="Number of clusters")
        self.num_clusters_label.pack()

        self.num_clusters_entry = tk.Entry(self.controls_frame)
        self.num_clusters_entry.insert(0, "10")
        self.num_clusters_entry.pack()

        # Radio buttons for scaling methods
        self.scaling_method = tk.StringVar()
        self.scaling_method.set("StandardScaler")
        self.scaling_methods = [
            ("StandardScaler", "StandardScaler"), 
            ("MinMaxScaler", "MinMaxScaler"), 
            ("RobustScaler", "RobustScaler"), 
            ("MaxAbsScaler", "MaxAbsScaler"),
        ]

        self.scaling_method_label = tk.Label(self.controls_frame, text="Scaling method")
        self.scaling_method_label.pack()

        for text, method in self.scaling_methods:
            b = tk.Radiobutton(self.controls_frame, text=text, variable=self.scaling_method, value=method)
            b.pack()

        # Create tab for elbow method and cluster visualization
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(side='right', fill='both', expand=True)

        # Elbow method tab
        self.elbow_tab = tk.Frame(self.notebook)
        self.notebook.add(self.elbow_tab, text="Elbow Method")

        self.elbow_canvas = FigureCanvasTkAgg(plt.figure(), self.elbow_tab)
        self.elbow_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Cluster visualization tab
        self.cluster_tab = tk.Frame(self.notebook)
        self.notebook.add(self.cluster_tab, text="Cluster Visualization")

        self.cluster_canvas = FigureCanvasTkAgg(plt.figure(), self.cluster_tab)
        self.cluster_canvas.get_tk_widget().pack(fill='both', expand=True)


    def on_closing(self):
        self.stop_event.set()
        if self.cluster_thread:
            self.cluster_thread.join(timeout=1)
            self.cluster_thread = None
        self.master.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpyGUI(root)
    root.mainloop()