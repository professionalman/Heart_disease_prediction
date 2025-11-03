import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import io


class HeartDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("❤️ Heart Disease Prediction Analysis")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f2f5")
        
        # Center window on screen
        self.center_window(1400, 900)

        # Center window on screen
        self.center_window(1400, 900)

        # Custom style configuration
        self.setup_styles()

        # Header Frame with gradient-like effect
        self.header_frame = ttk.Frame(root, style="Header.TFrame")
        self.header_frame.pack(fill="x", padx=0, pady=0)

        # Title with icon
        title_container = ttk.Frame(self.header_frame, style="Header.TFrame")
        title_container.pack(fill="x", pady=20)
        
        ttk.Label(title_container, text="❤️", 
                  font=("Segoe UI", 32), 
                  background="#e74c3c", 
                  foreground="white").pack(side="left", padx=(30, 10))
        
        title_frame = ttk.Frame(title_container, style="Header.TFrame")
        title_frame.pack(side="left")
        
        ttk.Label(title_frame, text="Heart Disease Prediction",
                  font=("Segoe UI", 20, "bold"),
                  background="#e74c3c",
                  foreground="white").pack(anchor="w")
        
        ttk.Label(title_frame, text="AI-Powered Medical Analysis System",
                  font=("Segoe UI", 10),
                  background="#e74c3c",
                  foreground="#ffebee").pack(anchor="w")

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root, style="Custom.TNotebook")
        self.notebook.pack(fill='both', expand=True, padx=20, pady=20)

        # Create tabs
        self.tab_data = ttk.Frame(self.notebook, style="TFrame")
        self.tab_visualization = ttk.Frame(self.notebook, style="TFrame")
        self.tab_model = ttk.Frame(self.notebook, style="TFrame")

        self.notebook.add(self.tab_data, text="📂 Data Explorer")
        self.notebook.add(self.tab_visualization, text="📊 Data Visualizer")
        self.notebook.add(self.tab_model, text="🧠 Model Trainer")

        # Initialize Variables
        self.dataset = None
        self.X = None
        self.y = None
        self.original_df = None

        # Setup UI Components
        self.setup_data_tab()
        self.setup_visualization_tab()
        self.setup_model_tab()

    def setup_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()
        style.theme_use('clam')  # Use clam theme for better customization

        # Main style configurations
        style.configure("TFrame", background="#f0f2f5")
        style.configure("TLabel", background="#f0f2f5", font=("Segoe UI", 10), foreground="#2c3e50")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=10, borderwidth=0)
        
        # Notebook styles
        style.configure("Custom.TNotebook", background="#f0f2f5", borderwidth=0, tabmargins=[10, 10, 10, 0])
        style.configure("Custom.TNotebook.Tab", 
                       font=("Segoe UI", 11, "bold"), 
                       padding=[20, 12],
                       background="#ecf0f1",
                       foreground="#34495e")
        style.map("Custom.TNotebook.Tab",
                 background=[("selected", "#ffffff"), ("active", "#e8eaf0")],
                 foreground=[("selected", "#e74c3c"), ("active", "#2c3e50")],
                 expand=[("selected", [1, 1, 1, 0])])

        # Custom header style
        style.configure("Header.TFrame", background="#e74c3c")
        style.configure("Header.TLabel", background="#e74c3c",
                        foreground="white", font=("Segoe UI", 14, "bold"))

        # Button styles with modern colors
        style.configure("Primary.TButton", 
                       background="#3498db", 
                       foreground="white",
                       relief="flat",
                       borderwidth=0)
        style.map("Primary.TButton",
                 background=[("active", "#2980b9"), ("pressed", "#21618c")],
                 relief=[("pressed", "flat"), ("active", "flat")])
        
        style.configure("Secondary.TButton", 
                       background="#95a5a6", 
                       foreground="white",
                       relief="flat")
        style.map("Secondary.TButton",
                 background=[("active", "#7f8c8d"), ("pressed", "#566573")])
        
        style.configure("Success.TButton", 
                       background="#27ae60", 
                       foreground="white",
                       relief="flat")
        style.map("Success.TButton",
                 background=[("active", "#229954"), ("pressed", "#1e8449")])

        # Info panel styles
        style.configure("Info.TFrame", background="#ffffff", relief="flat", borderwidth=1)
        style.configure("Info.TLabel", background="#ffffff", font=("Segoe UI", 10), foreground="#34495e")
        
        # Card style for sections
        style.configure("Card.TFrame", background="#ffffff", relief="flat", borderwidth=0)
        style.configure("Card.TLabel", background="#ffffff", font=("Segoe UI", 10), foreground="#2c3e50")
        
        # LabelFrame style
        style.configure("TLabelframe", background="#ffffff", relief="flat", borderwidth=2)
        style.configure("TLabelframe.Label", 
                       background="#ffffff", 
                       font=("Segoe UI", 11, "bold"), 
                       foreground="#2c3e50")
    
    def center_window(self, width, height):
        """Center the window on the screen"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def setup_data_tab(self):
        """Creates UI for the Data Tab"""
        # Main container frame
        container = ttk.Frame(self.tab_data, style="TFrame")
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Left panel - Actions
        left_panel = ttk.Frame(container, width=280, style="Card.TFrame")
        left_panel.pack(side="left", fill="y", padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Add shadow effect with border
        left_panel.configure(relief="solid", borderwidth=1)

        # Panel header
        header = ttk.Frame(left_panel, style="Card.TFrame")
        header.pack(fill="x", pady=(15, 10))
        
        ttk.Label(header, text="⚙️ Data Actions", 
                 font=("Segoe UI", 13, "bold"),
                 foreground="#2c3e50",
                 background="#ffffff").pack(pady=5)

        ttk.Button(left_panel, text="📂  Load Dataset",
                   command=self.load_dataset,
                   style="Primary.TButton").pack(fill="x", padx=15, pady=8)

        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=15, padx=10)

        # Stats section
        stats_header = ttk.Frame(left_panel, style="Card.TFrame")
        stats_header.pack(fill="x", padx=15)
        
        ttk.Label(stats_header, text="📊 Dataset Statistics", 
                 font=("Segoe UI", 11, "bold"),
                 foreground="#34495e",
                 background="#ffffff").pack(anchor="w", pady=(0, 10))

        self.stats_label = ttk.Label(left_panel, 
                                     text="No data loaded yet\n\nPlease load a CSV file\nto begin analysis", 
                                     style="Card.TLabel", 
                                     wraplength=240,
                                     justify="center",
                                     foreground="#7f8c8d")
        self.stats_label.pack(fill="x", padx=15, pady=10)

        # Right panel - Data display
        right_panel = ttk.Frame(container, style="Card.TFrame")
        right_panel.pack(side="right", fill="both", expand=True)
        right_panel.configure(relief="solid", borderwidth=1)

        # Dataset Info Section
        info_frame = ttk.LabelFrame(right_panel, text="  📋 Dataset Information  ", padding=20)
        info_frame.pack(fill="both", expand=True, padx=15, pady=15)

        self.info_text = tk.Text(info_frame, height=15, wrap="none", 
                                bg="#fafbfc",
                                font=("Consolas", 10), 
                                padx=10, pady=10,
                                relief="flat",
                                borderwidth=0,
                                fg="#2c3e50")
        self.info_text.pack(fill="both", expand=True, side="left")

        # Add Scrollbar
        scroll_y = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")

    def setup_visualization_tab(self):
        """Creates UI for the Visualization Tab"""
        # Main container frame
        container = ttk.Frame(self.tab_visualization, style="TFrame")
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Left panel - Visualization controls
        left_panel = ttk.Frame(container, width=280, style="Card.TFrame")
        left_panel.pack(side="left", fill="y", padx=(0, 15))
        left_panel.pack_propagate(False)
        left_panel.configure(relief="solid", borderwidth=1)

        # Panel header
        header = ttk.Frame(left_panel, style="Card.TFrame")
        header.pack(fill="x", pady=(15, 10))
        
        ttk.Label(header, text="📊 Visualization Tools", 
                 font=("Segoe UI", 13, "bold"),
                 foreground="#2c3e50",
                 background="#ffffff").pack(pady=5)

        ttk.Button(left_panel, text="🔥  Correlation Heatmap",
                   command=lambda: self.show_visualization("heatmap"),
                   style="Primary.TButton").pack(fill="x", padx=15, pady=8)

        ttk.Button(left_panel, text="📈  Feature Distribution",
                   command=lambda: self.show_visualization("hist"),
                   style="Primary.TButton").pack(fill="x", padx=15, pady=8)

        ttk.Button(left_panel, text="🎯  Target Distribution",
                   command=lambda: self.show_visualization("target"),
                   style="Primary.TButton").pack(fill="x", padx=15, pady=8)

        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=15, padx=10)

        info_label = ttk.Label(left_panel, 
                              text="💡 Tip:\n\nVisualize your data to\nunderstand patterns and\nrelationships between\nfeatures.",
                              style="Card.TLabel",
                              justify="center",
                              wraplength=240,
                              foreground="#7f8c8d")
        info_label.pack(padx=15, pady=10)

        # Right panel - Plot display
        right_panel = ttk.Frame(container, style="Card.TFrame")
        right_panel.pack(side="right", fill="both", expand=True)
        right_panel.configure(relief="solid", borderwidth=1)

        # Frame for Plots
        self.plot_frame = ttk.Frame(right_panel, style="Card.TFrame", padding=15)
        self.plot_frame.pack(fill="both", expand=True)

        # Placeholder for when no visualization is shown
        placeholder_container = ttk.Frame(self.plot_frame, style="Card.TFrame")
        placeholder_container.place(relx=0.5, rely=0.5, anchor="center")
        
        ttk.Label(placeholder_container,
                 text="📊",
                 font=("Segoe UI", 48),
                 background="#ffffff",
                 foreground="#bdc3c7").pack()
        
        self.placeholder_label = ttk.Label(placeholder_container,
                                          text="Select a visualization from the left panel\nto display charts and graphs",
                                          style="Card.TLabel",
                                          justify="center",
                                          foreground="#95a5a6")
        self.placeholder_label.pack(pady=10)

    def setup_model_tab(self):
        """Creates UI for Model Training Tab"""
        # Main container frame
        container = ttk.Frame(self.tab_model, style="TFrame")
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Left panel - Model controls
        left_panel = ttk.Frame(container, width=280, style="Card.TFrame")
        left_panel.pack(side="left", fill="y", padx=(0, 15))
        left_panel.pack_propagate(False)
        left_panel.configure(relief="solid", borderwidth=1)

        # Panel header
        header = ttk.Frame(left_panel, style="Card.TFrame")
        header.pack(fill="x", pady=(15, 10))
        
        ttk.Label(header, text="🧠 Model Training", 
                 font=("Segoe UI", 13, "bold"),
                 foreground="#2c3e50",
                 background="#ffffff").pack(pady=5)

        ttk.Button(left_panel, text="🔍  Evaluate KNN",
                   command=self.evaluate_knn,
                   style="Success.TButton").pack(fill="x", padx=15, pady=8)

        ttk.Button(left_panel, text="🌳  Evaluate Random Forest",
                   command=self.evaluate_rf,
                   style="Success.TButton").pack(fill="x", padx=15, pady=8)

        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=15, padx=10)

        info_label = ttk.Label(left_panel, 
                              text="🤖 Machine Learning\n\nTrain and evaluate\nmultiple classification\nmodels to predict\nheart disease.",
                              style="Card.TLabel",
                              justify="center",
                              wraplength=240,
                              foreground="#7f8c8d")
        info_label.pack(padx=15, pady=10)

        # Right panel - Results display
        right_panel = ttk.Frame(container, style="Card.TFrame")
        right_panel.pack(side="right", fill="both", expand=True)
        right_panel.configure(relief="solid", borderwidth=1)

        # Results Section
        results_frame = ttk.LabelFrame(right_panel, text="  📊 Model Evaluation Results  ", padding=20)
        results_frame.pack(fill="both", expand=True, padx=15, pady=15)

        self.results_text = tk.Text(results_frame, height=15, 
                                    bg="#fafbfc",
                                    font=("Consolas", 10), 
                                    padx=10, pady=10,
                                    relief="flat",
                                    borderwidth=0,
                                    fg="#2c3e50")
        self.results_text.pack(fill="both", expand=True, side="left")

        # Add Scrollbar for Results
        scroll_y = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")

    def load_dataset(self):
        """Loads the CSV dataset"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.original_df = pd.read_csv(file_path)
                self.dataset = self.original_df.copy()
                self.preprocess_data()

                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, f"Shape: {self.original_df.shape}\n\n")

                buffer = io.StringIO()
                self.original_df.info(buf=buffer)
                self.info_text.insert(tk.END, buffer.getvalue())

                # Update stats label
                self.stats_label.config(
                    text=f"✅ Dataset Loaded\n\n📊 Rows: {self.original_df.shape[0]}\n📈 Features: {self.original_df.shape[1]}\n\n🎯 Ready for Analysis!",
                    foreground="#27ae60",
                    font=("Segoe UI", 10, "bold"))

                # Clear any existing visualizations
                for widget in self.plot_frame.winfo_children():
                    widget.destroy()
                
                placeholder_container = ttk.Frame(self.plot_frame, style="Card.TFrame")
                placeholder_container.place(relx=0.5, rely=0.5, anchor="center")
                
                ttk.Label(placeholder_container,
                         text="📊",
                         font=("Segoe UI", 48),
                         background="#ffffff",
                         foreground="#bdc3c7").pack()
                
                self.placeholder_label = ttk.Label(placeholder_container,
                                                  text="Select a visualization from the left panel\nto display charts and graphs",
                                                  style="Card.TLabel",
                                                  justify="center",
                                                  foreground="#95a5a6")
                self.placeholder_label.pack(pady=10)

                messagebox.showinfo("✅ Success", "Dataset loaded successfully!\n\nYou can now explore visualizations and train models.")
            except Exception as e:
                messagebox.showerror("❌ Error", f"Failed to load dataset: {str(e)}")

    def preprocess_data(self):
        if self.original_df is not None:
            self.dataset = self.original_df.copy()

            # Ensure the target variable is categorical
            if 'target' in self.dataset.columns:
                self.dataset['target'] = self.dataset['target'].astype(int)  # Convert to integer

            # Normalize numerical features (excluding target)
            standardScaler = StandardScaler()
            numeric_cols = [col for col in self.dataset.columns if
                            col != 'target' and self.dataset[col].dtype in ['float64', 'int64']]
            self.dataset[numeric_cols] = standardScaler.fit_transform(self.dataset[numeric_cols])

            self.y = self.dataset['target']  # Now correctly categorized
            self.X = self.dataset.drop(['target'], axis=1)  # Features

    def show_visualization(self, vis_type):
        """Displays plots for data visualization"""
        if self.original_df is None:
            messagebox.showwarning("⚠️ Warning", "Please load a dataset first!")
            return

        # Clear the plot frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Set modern style for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#ffffff')

        if vis_type == "hist":
            fig.clear()
            cols = self.original_df.select_dtypes(include=['int64', 'float64']).columns[:9]
            rows = int(np.ceil(len(cols) / 3))
            axes = fig.subplots(rows, 3)

            if rows == 1:
                axes = np.array([axes])

            for i, col in enumerate(cols):
                row_idx, col_idx = divmod(i, 3)
                ax = axes[row_idx, col_idx]
                ax.hist(self.original_df[col], bins=15, color='#3498db', edgecolor='#2c3e50', alpha=0.7)
                ax.set_title(col, fontsize=10, fontweight='bold', color='#2c3e50')
                ax.set_facecolor('#f8f9fa')
                ax.grid(True, alpha=0.3)

            for i in range(len(cols), rows * 3):
                fig.delaxes(axes.flat[i])

            fig.suptitle('Feature Distribution Analysis', fontsize=14, fontweight='bold', color='#2c3e50', y=0.995)
            fig.tight_layout()

        elif vis_type == "target":
            ax = fig.add_subplot(111)
            colors = ['#e74c3c', '#27ae60']
            counts = self.original_df['target'].value_counts()
            bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='#2c3e50', linewidth=1.5, alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            ax.set_xlabel('Target', fontsize=12, fontweight='bold', color='#2c3e50')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold', color='#2c3e50')
            ax.set_title("Target Variable Distribution\n(0: No Disease, 1: Disease)", 
                        fontsize=13, fontweight='bold', color='#2c3e50', pad=20)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No Disease (0)', 'Disease (1)'])
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, alpha=0.3, axis='y')

        elif vis_type == "heatmap":
            ax = fig.add_subplot(111)
            corr = self.original_df.corr()
            sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, 
                       fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8},
                       ax=ax, square=True)
            ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold', 
                        color='#2c3e50', pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)

        # Embed the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def evaluate_knn(self):
        if self.X is None or self.y is None:
            messagebox.showwarning("⚠️ Warning", "Please load and preprocess a dataset first!")
            return

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🔍 Evaluating KNN Classifier...\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        self.root.update()

        knn_scores = []
        for k in range(1, 21):
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn_classifier, self.X, self.y, cv=10)
            knn_scores.append(score.mean())
            self.results_text.insert(tk.END, f"K = {k:2d}  →  Score: {score.mean():.4f}\n")
            self.root.update()

        # **Fix: Use self.plot_frame instead of self.knn_plot_frame**
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 6))
        fig.patch.set_facecolor('#ffffff')
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        ax = fig.add_subplot(111)
        ax.plot([k for k in range(1, 21)], knn_scores, color='#e74c3c', 
               linewidth=2.5, marker='o', markersize=6, markerfacecolor='#c0392b')
        
        # Highlight best K
        best_k = np.argmax(knn_scores) + 1
        ax.plot(best_k, knn_scores[best_k-1], 'go', markersize=12, 
               label=f'Best K={best_k}', zorder=5)
        
        for i in range(1, 21):
            if i % 2 == 1:  # Show labels for odd K values to avoid clutter
                ax.text(i, knn_scores[i - 1] + 0.005, f'{knn_scores[i - 1]:.3f}', 
                       ha='center', va='bottom', fontsize=8, color='#2c3e50')
        
        ax.set_xticks([i for i in range(1, 21)])
        ax.set_xlabel('Number of Neighbors (K)', fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Cross-Validation Score', fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_title('KNN Classifier Performance Analysis', fontsize=13, fontweight='bold', 
                    color='#2c3e50', pad=20)
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        fig.tight_layout()

        canvas.draw()

        self.results_text.insert(tk.END, f"\n{'=' * 50}\n")
        self.results_text.insert(tk.END, f"✅ Best K value: {best_k}\n")
        self.results_text.insert(tk.END, f"🎯 Best Score: {max(knn_scores):.4f}\n")
        self.results_text.insert(tk.END, f"{'=' * 50}\n")

    def evaluate_rf(self):
        if self.X is None or self.y is None:
            messagebox.showwarning("⚠️ Warning", "Please load and preprocess a dataset first!")
            return

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🌳 Evaluating Random Forest Classifier...\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        self.root.update()

        randomforest_classifier = RandomForestClassifier(n_estimators=10)
        score = cross_val_score(randomforest_classifier, self.X, self.y, cv=10).mean()

        self.results_text.insert(tk.END, f"📊 Cross-Validation Results (10-fold):\n\n")
        self.results_text.insert(tk.END, f"   Algorithm: Random Forest\n")
        self.results_text.insert(tk.END, f"   Estimators: 10 trees\n")
        self.results_text.insert(tk.END, f"   Average Score: {score:.4f}\n")
        self.results_text.insert(tk.END, f"   Accuracy: {score*100:.2f}%\n\n")
        self.results_text.insert(tk.END, f"{'=' * 50}\n")
        self.results_text.insert(tk.END, f"✅ Model Evaluation Complete!\n")
        self.results_text.insert(tk.END, f"{'=' * 50}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseApp(root)
    root.mainloop()