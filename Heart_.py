import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# Design tokens
C = {
    "bg": "#F4F6FA",
    "sidebar": "#0F172A",
    "sidebar_hover": "#1E293B",
    "sidebar_text": "#94A3B8",
    "card": "#FFFFFF",
    "subtle": "#F8FAFC",
    "border": "#E2E8F0",
    "text": "#0F172A",
    "muted": "#64748B",
    "faint": "#CBD5E1",
    "accent": "#E11D48",
    "accent_hover": "#BE123C",
    "accent_soft": "#FFE4E6",
    "green": "#059669",
    "green_hover": "#047857",
    "green_soft": "#D1FAE5",
    "healthy": "#14B8A6",
}
FONT = "Segoe UI"
MONO = "Consolas"
AUTHOR = "Kavya Arora"

matplotlib.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

BUTTON_KINDS = {
    # kind: (bg, hover bg, fg, border)
    "primary": (C["accent"], C["accent_hover"], "#FFFFFF", C["accent"]),
    "success": (C["green"], C["green_hover"], "#FFFFFF", C["green"]),
    "ghost": (C["card"], C["subtle"], C["text"], C["border"]),
}

TARGET_LABELS = {0: "No disease", 1: "Disease"}
TARGET_COLORS = {0: C["healthy"], 1: C["accent"]}


class HeartDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Heart Disease Prediction — by {AUTHOR}")
        self.root.configure(bg=C["bg"])
        self.root.minsize(1100, 720)
        self.center_window(1400, 900)

        self.setup_styles()

        # Initialize Variables
        self.dataset = None
        self.X = None
        self.y = None
        self.original_df = None
        self.dataset_name = None

        self.pages = {}
        self.nav_items = {}
        self.page_meta = {
            "data": ("Data Explorer", "Inspect the raw dataset, column types and summary statistics"),
            "viz": ("Data Visualizer", "Explore feature relationships and class balance"),
            "model": ("Model Trainer", "Evaluate classifiers with 10-fold cross-validation"),
        }

        self.build_layout()
        self.setup_data_tab()
        self.setup_visualization_tab()
        self.setup_model_tab()
        self.show_page("data")

    # ------------------------------------------------------------------ styles
    def setup_styles(self):
        """Configure ttk styles for tables and scrollbars"""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure("Data.Treeview",
                        background=C["card"], fieldbackground=C["card"],
                        foreground=C["text"], font=(FONT, 9), rowheight=28,
                        borderwidth=0, relief="flat")
        style.map("Data.Treeview",
                  background=[("selected", C["accent_soft"])],
                  foreground=[("selected", C["text"])])
        style.configure("Data.Treeview.Heading",
                        background=C["subtle"], foreground=C["muted"],
                        font=(FONT, 9, "bold"), relief="flat", borderwidth=0, padding=(8, 8))
        style.map("Data.Treeview.Heading", background=[("active", C["border"])])
        style.layout("Data.Treeview", [("Data.Treeview.treearea", {"sticky": "nswe"})])

        for orient in ("Vertical", "Horizontal"):
            style.configure(f"Slim.{orient}.TScrollbar",
                            background=C["faint"], troughcolor=C["card"],
                            bordercolor=C["card"], lightcolor=C["faint"], darkcolor=C["faint"],
                            arrowcolor=C["muted"], gripcount=0, arrowsize=10)
            style.map(f"Slim.{orient}.TScrollbar", background=[("active", C["muted"])])

    def center_window(self, width, height):
        """Center the window on the screen"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width, height = min(width, screen_width - 40), min(height, screen_height - 80)
        x = (screen_width - width) // 2
        y = max((screen_height - height) // 2 - 20, 0)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    # ----------------------------------------------------------- UI helpers
    def make_button(self, parent, text, command, kind="primary", **pack_opts):
        """Flat button with hover state, built from a Label for full color control"""
        bg, hover, fg, border = BUTTON_KINDS[kind]
        btn = tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, 10, "bold"),
                       padx=16, pady=9, cursor="hand2",
                       highlightthickness=1, highlightbackground=border)
        btn.bind("<Enter>", lambda e: btn.config(bg=hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        btn.bind("<Button-1>", lambda e: command())
        return btn

    def make_card(self, parent, title=None, subtitle=None, padding=18):
        """White bordered card. Returns (card, body)"""
        card = tk.Frame(parent, bg=C["card"], highlightthickness=1, highlightbackground=C["border"])
        if title:
            head = tk.Frame(card, bg=C["card"])
            head.pack(fill="x", padx=padding, pady=(padding, 0))
            tk.Label(head, text=title, bg=C["card"], fg=C["text"],
                     font=(FONT, 12, "bold")).pack(anchor="w")
            if subtitle:
                tk.Label(head, text=subtitle, bg=C["card"], fg=C["muted"],
                         font=(FONT, 9)).pack(anchor="w", pady=(2, 0))
        body = tk.Frame(card, bg=C["card"])
        body.pack(fill="both", expand=True, padx=padding, pady=padding)
        return card, body

    def make_table(self, parent):
        """Treeview with slim scrollbars. Returns (container, tree)"""
        # Fixed small request so wide tables scroll instead of stretching the window
        container = tk.Frame(parent, bg=C["card"], highlightthickness=1, highlightbackground=C["border"],
                             width=200, height=200)
        container.grid_propagate(False)
        tree = ttk.Treeview(container, style="Data.Treeview", show="headings")
        ys = ttk.Scrollbar(container, orient="vertical", command=tree.yview, style="Slim.Vertical.TScrollbar")
        xs = ttk.Scrollbar(container, orient="horizontal", command=tree.xview, style="Slim.Horizontal.TScrollbar")
        tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        tree.tag_configure("odd", background=C["subtle"])
        return container, tree

    def show_empty_state(self, frame, icon, title, message):
        """Centered placeholder for areas with nothing to show yet"""
        holder = tk.Frame(frame, bg=C["card"])
        holder.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(holder, text=icon, bg=C["accent_soft"], fg=C["accent"],
                 font=(FONT, 22), width=3).pack(pady=(0, 12))
        tk.Label(holder, text=title, bg=C["card"], fg=C["text"],
                 font=(FONT, 12, "bold")).pack()
        tk.Label(holder, text=message, bg=C["card"], fg=C["muted"],
                 font=(FONT, 10), justify="center").pack(pady=(4, 0))
        return holder

    def embed_figure(self, canvas):
        """Fill the parent frame; place() ignores the figure's requested size so it always fits"""
        widget = canvas.get_tk_widget()
        widget.configure(highlightthickness=0)
        widget.place(x=0, y=0, relwidth=1, relheight=1)

    def clear(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def set_status(self, text, color=None):
        self.status_label.config(text=text, fg=color or C["muted"])

    # ----------------------------------------------------------------- layout
    def build_layout(self):
        """Sidebar navigation, page header, content area and status bar"""
        # Sidebar
        sidebar = tk.Frame(self.root, bg=C["sidebar"], width=240)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        brand = tk.Frame(sidebar, bg=C["sidebar"])
        brand.pack(fill="x", padx=22, pady=(26, 30))
        tk.Label(brand, text="♥", bg=C["accent"], fg="white",
                 font=(FONT, 16, "bold"), width=2).pack(side="left")
        brand_text = tk.Frame(brand, bg=C["sidebar"])
        brand_text.pack(side="left", padx=(12, 0))
        tk.Label(brand_text, text="CardioPredict", bg=C["sidebar"], fg="white",
                 font=(FONT, 13, "bold")).pack(anchor="w")
        tk.Label(brand_text, text="Heart disease analysis", bg=C["sidebar"],
                 fg=C["sidebar_text"], font=(FONT, 8)).pack(anchor="w")

        tk.Label(sidebar, text="WORKSPACE", bg=C["sidebar"], fg="#475569",
                 font=(FONT, 8, "bold")).pack(anchor="w", padx=24, pady=(0, 8))

        for key, icon, label in (("data", "▦", "Data Explorer"),
                                 ("viz", "◔", "Data Visualizer"),
                                 ("model", "⚙", "Model Trainer")):
            self.add_nav_item(sidebar, key, icon, label)

        # Author credit (packed first so it sits at the very bottom)
        credit = tk.Frame(sidebar, bg=C["sidebar"])
        credit.pack(side="bottom", fill="x", padx=30, pady=(0, 20))
        tk.Label(credit, text="DEVELOPED BY", bg=C["sidebar"], fg="#475569",
                 font=(FONT, 8, "bold")).pack(anchor="w")
        tk.Label(credit, text=f"♥  {AUTHOR}", bg=C["sidebar"], fg="white",
                 font=(FONT, 11, "bold")).pack(anchor="w", pady=(2, 0))

        # Sidebar footer: dataset status
        footer = tk.Frame(sidebar, bg=C["sidebar_hover"])
        footer.pack(side="bottom", fill="x", padx=16, pady=(18, 14))
        tk.Label(footer, text="DATASET", bg=C["sidebar_hover"], fg="#475569",
                 font=(FONT, 8, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
        self.sidebar_dataset = tk.Label(footer, text="Nothing loaded", bg=C["sidebar_hover"],
                                        fg="white", font=(FONT, 10, "bold"), anchor="w")
        self.sidebar_dataset.pack(fill="x", padx=14)
        self.sidebar_dataset_meta = tk.Label(footer, text="Load a CSV to begin", bg=C["sidebar_hover"],
                                             fg=C["sidebar_text"], font=(FONT, 9), anchor="w")
        self.sidebar_dataset_meta.pack(fill="x", padx=14, pady=(0, 12))

        # Main area
        main = tk.Frame(self.root, bg=C["bg"])
        main.pack(side="left", fill="both", expand=True)

        header = tk.Frame(main, bg=C["bg"])
        header.pack(fill="x", padx=32, pady=(26, 18))
        titles = tk.Frame(header, bg=C["bg"])
        titles.pack(side="left")
        self.page_title = tk.Label(titles, bg=C["bg"], fg=C["text"], font=(FONT, 20, "bold"))
        self.page_title.pack(anchor="w")
        self.page_subtitle = tk.Label(titles, bg=C["bg"], fg=C["muted"], font=(FONT, 10))
        self.page_subtitle.pack(anchor="w")
        self.make_button(header, "＋  Load dataset", self.load_dataset).pack(side="right")

        status_bar = tk.Frame(main, bg=C["card"], highlightthickness=1, highlightbackground=C["border"])
        status_bar.pack(side="bottom", fill="x")
        self.status_label = tk.Label(status_bar, text="Ready", bg=C["card"], fg=C["muted"],
                                     font=(FONT, 9), anchor="w")
        tk.Label(status_bar, text=f"© {AUTHOR}", bg=C["card"], fg=C["muted"],
                 font=(FONT, 9, "bold")).pack(side="right", padx=32, pady=6)
        self.status_label.pack(side="left", fill="x", expand=True, padx=32, pady=6)

        self.content = tk.Frame(main, bg=C["bg"])
        self.content.pack(fill="both", expand=True, padx=32, pady=(0, 24))
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

        for key in self.page_meta:
            page = tk.Frame(self.content, bg=C["bg"])
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page

        self.tab_data = self.pages["data"]
        self.tab_visualization = self.pages["viz"]
        self.tab_model = self.pages["model"]

    def add_nav_item(self, sidebar, key, icon, label):
        row = tk.Frame(sidebar, bg=C["sidebar"], cursor="hand2")
        row.pack(fill="x", padx=12, pady=2)
        bar = tk.Frame(row, bg=C["sidebar"], width=3)
        bar.pack(side="left", fill="y")
        icon_lbl = tk.Label(row, text=icon, bg=C["sidebar"], fg=C["sidebar_text"],
                            font=(FONT, 12), width=2)
        icon_lbl.pack(side="left", padx=(10, 6), pady=9)
        text_lbl = tk.Label(row, text=label, bg=C["sidebar"], fg=C["sidebar_text"],
                            font=(FONT, 10, "bold"), anchor="w")
        text_lbl.pack(side="left", fill="x", expand=True)

        widgets = (row, icon_lbl, text_lbl)
        self.nav_items[key] = {"widgets": widgets, "bar": bar, "icon": icon_lbl, "text": text_lbl}

        def on_enter(_):
            if self.current_page != key:
                for w in widgets:
                    w.config(bg=C["sidebar_hover"])

        def on_leave(_):
            if self.current_page != key:
                for w in widgets:
                    w.config(bg=C["sidebar"])

        for w in widgets:
            w.bind("<Button-1>", lambda e: self.show_page(key))
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)

    current_page = None

    def show_page(self, key):
        self.current_page = key
        self.pages[key].tkraise()
        title, subtitle = self.page_meta[key]
        self.page_title.config(text=title)
        self.page_subtitle.config(text=subtitle)
        for k, item in self.nav_items.items():
            active = k == key
            bg = C["sidebar_hover"] if active else C["sidebar"]
            for w in item["widgets"]:
                w.config(bg=bg)
            item["bar"].config(bg=C["accent"] if active else C["sidebar"])
            item["icon"].config(fg=C["accent"] if active else C["sidebar_text"])
            item["text"].config(fg="white" if active else C["sidebar_text"])

    # -------------------------------------------------------------- data page
    def setup_data_tab(self):
        """Creates UI for the Data Tab"""
        page = self.tab_data

        # KPI cards
        kpi_row = tk.Frame(page, bg=C["bg"])
        kpi_row.pack(fill="x", pady=(0, 18))
        self.kpis = {}
        kpi_defs = (("rows", "Patients", "records in dataset", C["accent"]),
                    ("features", "Features", "input columns", "#6366F1"),
                    ("positive", "Disease rate", "patients with target = 1", "#F59E0B"),
                    ("missing", "Missing values", "empty cells", C["healthy"]))
        for i, (key, title, caption, color) in enumerate(kpi_defs):
            kpi_row.columnconfigure(i, weight=1, uniform="kpi")
            card = tk.Frame(kpi_row, bg=C["card"], highlightthickness=1, highlightbackground=C["border"])
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 9, 0 if i == 3 else 9))
            tk.Frame(card, bg=color, height=3).pack(fill="x")
            inner = tk.Frame(card, bg=C["card"])
            inner.pack(fill="both", padx=18, pady=14)
            tk.Label(inner, text=title.upper(), bg=C["card"], fg=C["muted"],
                     font=(FONT, 8, "bold")).pack(anchor="w")
            value = tk.Label(inner, text="—", bg=C["card"], fg=C["text"], font=(FONT, 22, "bold"))
            value.pack(anchor="w", pady=(2, 0))
            tk.Label(inner, text=caption, bg=C["card"], fg=C["muted"], font=(FONT, 9)).pack(anchor="w")
            self.kpis[key] = value

        # Tables
        grid = tk.Frame(page, bg=C["bg"])
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=3, uniform="tables")
        grid.columnconfigure(1, weight=2, uniform="tables")
        grid.rowconfigure(0, weight=1)

        preview_card, preview_body = self.make_card(grid, "Data preview", "First 200 rows of the loaded file")
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        table, self.preview_tree = self.make_table(preview_body)
        table.pack(fill="both", expand=True)
        self.preview_empty = self.show_empty_state(
            preview_body, "▦", "No dataset loaded",
            "Click “Load dataset” to open a CSV file\n(heart.csv is included with this project)")

        summary_card, summary_body = self.make_card(grid, "Column summary", "Type, completeness and range per column")
        summary_card.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        table, self.summary_tree = self.make_table(summary_body)
        table.pack(fill="both", expand=True)
        self.summary_empty = self.show_empty_state(summary_body, "≡", "Nothing to summarize",
                                                   "Column details appear here")

    def populate_data_tab(self):
        df = self.original_df

        self.kpis["rows"].config(text=f"{len(df):,}")
        self.kpis["features"].config(text=str(df.shape[1] - (1 if 'target' in df.columns else 0)))
        self.kpis["positive"].config(
            text=f"{df['target'].mean() * 100:.1f}%" if 'target' in df.columns else "n/a")
        self.kpis["missing"].config(text=f"{int(df.isna().sum().sum()):,}")

        # Preview
        self.preview_empty.place_forget()
        tree = self.preview_tree
        tree.delete(*tree.get_children())
        cols = list(df.columns)
        tree["columns"] = cols
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=78, minwidth=60, anchor="center", stretch=True)
        for i, row in enumerate(df.head(200).itertuples(index=False)):
            values = [f"{v:g}" if isinstance(v, (float, np.floating)) else v for v in row]
            tree.insert("", "end", values=values, tags=("odd",) if i % 2 else ())

        # Summary
        self.summary_empty.place_forget()
        tree = self.summary_tree
        tree.delete(*tree.get_children())
        headers = ("Column", "Type", "Count", "Mean", "Min", "Max")
        tree["columns"] = headers
        widths = {"Column": 80, "Type": 58, "Count": 62}
        for h in headers:
            tree.heading(h, text=h)
            tree.column(h, width=widths.get(h, 50), minwidth=45,
                        anchor="w" if h == "Column" else "center")
        for i, col in enumerate(cols):
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                stats = (f"{s.mean():.2f}", f"{s.min():g}", f"{s.max():g}")
            else:
                stats = ("—", "—", "—")
            tree.insert("", "end", tags=("odd",) if i % 2 else (),
                        values=(col, str(s.dtype), f"{s.notna().sum():,}", *stats))

    def load_dataset(self):
        """Loads the CSV dataset"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            initialdir=os.path.dirname(os.path.abspath(__file__)))
        if file_path:
            try:
                self.original_df = pd.read_csv(file_path)
                self.dataset = self.original_df.copy()
                self.preprocess_data()
                self.dataset_name = os.path.basename(file_path)

                self.populate_data_tab()
                self.sidebar_dataset.config(text=self.dataset_name)
                self.sidebar_dataset_meta.config(
                    text=f"{self.original_df.shape[0]:,} rows · {self.original_df.shape[1]} columns")

                # Reset visualizations and model results for the new data
                self.clear(self.plot_frame)
                self.show_empty_state(self.plot_frame, "◔", "Pick a chart",
                                      "Choose a visualization above to render it here")
                self.reset_model_results()

                self.set_status(f"✓  Loaded {self.dataset_name} — ready to visualize and train", C["green"])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
                self.set_status("Failed to load dataset", C["accent"])

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

    # ------------------------------------------------------- visualizer page
    def setup_visualization_tab(self):
        """Creates UI for the Visualization Tab"""
        page = self.tab_visualization

        toolbar = tk.Frame(page, bg=C["bg"])
        toolbar.pack(fill="x", pady=(0, 18))

        segmented = tk.Frame(toolbar, bg=C["border"], padx=3, pady=3)
        segmented.pack(side="left")
        self.viz_buttons = {}
        for key, label in (("heatmap", "Correlation heatmap"),
                           ("hist", "Feature distribution"),
                           ("target", "Target distribution")):
            btn = tk.Label(segmented, text=label, bg=C["border"], fg=C["muted"],
                           font=(FONT, 10, "bold"), padx=16, pady=7, cursor="hand2")
            btn.pack(side="left", padx=1)
            btn.bind("<Button-1>", lambda e, k=key: self.show_visualization(k))
            self.viz_buttons[key] = btn

        self.viz_caption = tk.Label(toolbar, text="", bg=C["bg"], fg=C["muted"], font=(FONT, 9))
        self.viz_caption.pack(side="left", padx=16)

        card, self.plot_frame = self.make_card(page, padding=12)
        card.pack(fill="both", expand=True)
        self.show_empty_state(self.plot_frame, "◔", "No chart yet",
                              "Load a dataset, then choose a visualization above")

    def style_axes(self, ax, grid_axis="y"):
        ax.set_facecolor(C["card"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C["border"])
        ax.tick_params(colors=C["muted"], labelsize=9, length=0, pad=6)
        ax.grid(True, axis=grid_axis, color=C["border"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.title.set_color(C["text"])

    def show_visualization(self, vis_type):
        """Displays plots for data visualization"""
        if self.original_df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return

        for key, btn in self.viz_buttons.items():
            active = key == vis_type
            btn.config(bg=C["card"] if active else C["border"], fg=C["accent"] if active else C["muted"])

        self.clear(self.plot_frame)
        df = self.original_df
        fig = Figure(figsize=(6, 4), dpi=100, facecolor=C["card"], layout="constrained")

        if vis_type == "hist":
            features = [c for c in df.select_dtypes(include='number').columns if c != 'target']
            ncols = 4
            nrows = int(np.ceil(len(features) / ncols))
            axes = fig.subplots(nrows, ncols, squeeze=False)

            for ax, col in zip(axes.flat, features):
                bins = np.histogram_bin_edges(df[col].dropna(), bins=15)
                if 'target' in df.columns:
                    for val in (0, 1):
                        ax.hist(df.loc[df['target'] == val, col], bins=bins, alpha=0.6,
                                color=TARGET_COLORS[val], edgecolor="white", linewidth=0.5)
                else:
                    ax.hist(df[col], bins=bins, color=C["accent"], edgecolor="white", linewidth=0.5)
                self.style_axes(ax)
                ax.set_title(col, fontsize=10, fontweight='bold', loc="left")
                ax.tick_params(labelsize=7)

            for ax in axes.flat[len(features):]:
                fig.delaxes(ax)

            if 'target' in df.columns:
                handles = [Line2D([], [], marker="s", linestyle="", markersize=9, color=TARGET_COLORS[v])
                           for v in (0, 1)]
                fig.legend(handles, [TARGET_LABELS[v] for v in (0, 1)], loc="upper right",
                           ncol=2, frameon=False, fontsize=9)
            fig.suptitle('Feature distributions by outcome', fontsize=13, fontweight='bold',
                         color=C["text"], x=0.01, ha="left")
            caption = "Overlaid histograms for patients with and without heart disease"

        elif vis_type == "target":
            ax = fig.add_subplot(111)
            counts = df['target'].value_counts().sort_index()
            labels = [TARGET_LABELS.get(v, str(v)) for v in counts.index]
            colors = [TARGET_COLORS.get(v, C["muted"]) for v in counts.index]
            bars = ax.bar(labels, counts.values, color=colors, width=0.5)

            total = counts.sum()
            for bar, n in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{n:,}\n{n / total:.1%}", ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color=C["text"], linespacing=1.4)

            self.style_axes(ax)
            ax.spines["left"].set_visible(False)
            ax.set_ylim(0, counts.max() * 1.2)
            ax.set_ylabel('Patients', fontsize=10, color=C["muted"])
            ax.tick_params(axis="x", labelsize=11, labelcolor=C["text"])
            ax.set_title("Class balance of the target variable", fontsize=13,
                         fontweight='bold', loc="left", pad=16)
            caption = "How many patients fall into each outcome class"

        elif vis_type == "heatmap":
            ax = fig.add_subplot(111)
            # Lower triangle only: drop the empty first row / last column
            corr = df.select_dtypes(include='number').corr().iloc[1:, :-1]
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', vmin=-1, vmax=1, center=0,
                        cmap=sns.diverging_palette(190, 350, s=80, l=55, as_cmap=True),
                        linewidths=1.5, linecolor=C["card"], square=True,
                        annot_kws={"size": 7}, cbar_kws={'shrink': 0.7}, ax=ax)
            ax.set_title("Feature correlation matrix", fontsize=13, fontweight='bold',
                         color=C["text"], loc="left", pad=16)
            ax.tick_params(colors=C["muted"], labelsize=9, length=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            caption = "Pearson correlation between every pair of columns"

        self.viz_caption.config(text=caption)

        # Embed the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        self.embed_figure(canvas)

    # ------------------------------------------------------------ model page
    def setup_model_tab(self):
        """Creates UI for Model Training Tab"""
        page = self.tab_model
        page.columnconfigure(0, weight=0, minsize=300)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=3)
        page.rowconfigure(1, weight=2)

        # Left column: controls + metrics
        left = tk.Frame(page, bg=C["bg"], width=300)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 18))

        controls, body = self.make_card(left, "Classifiers", "10-fold CV on standardized features")
        controls.pack(fill="x")
        self.make_button(body, "Run K-Nearest Neighbors", self.evaluate_knn, "success").pack(fill="x")
        tk.Label(body, text="Sweeps k from 1 to 20 and picks the best", bg=C["card"],
                 fg=C["muted"], font=(FONT, 9)).pack(anchor="w", pady=(4, 12))
        self.make_button(body, "Run Random Forest", self.evaluate_rf, "ghost").pack(fill="x")
        tk.Label(body, text="Ensemble of 10 decision trees", bg=C["card"],
                 fg=C["muted"], font=(FONT, 9)).pack(anchor="w", pady=(4, 0))

        metrics, body = self.make_card(left, "Results")
        metrics.pack(fill="x", pady=(18, 0))
        self.metrics = {}
        for i, (key, label) in enumerate((("knn", "KNN accuracy"),
                                          ("best_k", "Best k"),
                                          ("rf", "Random Forest accuracy"))):
            if i:
                tk.Frame(body, bg=C["border"], height=1).pack(fill="x", pady=10)
            row = tk.Frame(body, bg=C["card"])
            row.pack(fill="x")
            tk.Label(row, text=label, bg=C["card"], fg=C["muted"], font=(FONT, 10)).pack(side="left")
            value = tk.Label(row, text="—", bg=C["card"], fg=C["text"], font=(FONT, 15, "bold"))
            value.pack(side="right")
            self.metrics[key] = value

        # Right column: chart + log
        chart_card, self.model_plot_frame = self.make_card(page, padding=12)
        chart_card.grid(row=0, column=1, sticky="nsew", pady=(0, 18))

        log_card, body = self.make_card(page, "Evaluation log")
        log_card.grid(row=1, column=1, sticky="nsew")
        self.results_text = tk.Text(body, height=8, bg=C["subtle"], fg=C["text"],
                                    font=(MONO, 10), padx=12, pady=10, relief="flat",
                                    borderwidth=0, highlightthickness=0, wrap="word")
        scroll_y = ttk.Scrollbar(body, orient="vertical", command=self.results_text.yview,
                                 style="Slim.Vertical.TScrollbar")
        self.results_text.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        self.results_text.pack(fill="both", expand=True, side="left")
        self.results_text.tag_configure("head", font=(MONO, 10, "bold"), foreground=C["text"])
        self.results_text.tag_configure("muted", foreground=C["muted"])
        self.results_text.tag_configure("good", font=(MONO, 10, "bold"), foreground=C["green"])

        self.reset_model_results()

    def reset_model_results(self):
        for value in self.metrics.values():
            value.config(text="—", fg=C["text"])
        self.clear(self.model_plot_frame)
        self.show_empty_state(self.model_plot_frame, "⚙", "No model trained yet",
                              "Run a classifier to see its performance curve")
        self.results_text.delete(1.0, tk.END)
        self.log("Waiting for a model run…\n", "muted")

    def log(self, text, tag=None):
        self.results_text.insert(tk.END, text, tag)
        self.results_text.see(tk.END)

    def begin_run(self, title):
        if self.X is None or self.y is None:
            messagebox.showwarning("Warning", "Please load and preprocess a dataset first!")
            return False
        self.show_page("model")
        self.results_text.delete(1.0, tk.END)
        self.log(f"{title}\n", "head")
        self.log(f"{self.dataset_name} · {len(self.X):,} samples · {self.X.shape[1]} features\n\n", "muted")
        self.set_status(f"Running {title.lower()}…", C["text"])
        self.root.config(cursor="watch")
        self.root.update()
        return True

    def end_run(self, message):
        self.root.config(cursor="")
        self.set_status(f"✓  {message}", C["green"])

    def evaluate_knn(self):
        if not self.begin_run("K-Nearest Neighbors evaluation"):
            return

        knn_scores = []
        for k in range(1, 21):
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn_classifier, self.X, self.y, cv=10)
            knn_scores.append(score.mean())
            self.log(f"  k = {k:2d}   accuracy {score.mean():.4f}\n")
            self.root.update()

        best_k = int(np.argmax(knn_scores)) + 1
        best_score = knn_scores[best_k - 1]

        self.clear(self.model_plot_frame)
        fig = Figure(figsize=(6, 4), dpi=100, facecolor=C["card"], layout="constrained")
        ax = fig.add_subplot(111)
        ks = list(range(1, 21))
        ax.fill_between(ks, knn_scores, min(knn_scores) - 0.01, color=C["accent"], alpha=0.08)
        ax.plot(ks, knn_scores, color=C["accent"], linewidth=2.2, marker='o', markersize=5,
                markerfacecolor=C["card"], markeredgewidth=1.8)
        ax.plot(best_k, best_score, 'o', markersize=11, color=C["green"], zorder=5)
        ax.text(0.99, 0.96, f"● Best k = {best_k}  ·  {best_score:.1%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, fontweight="bold", color=C["green"])

        self.style_axes(ax, grid_axis="both")
        ax.set_xticks(ks)
        ax.set_ylim(min(knn_scores) - 0.01, max(knn_scores) + 0.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel('Number of neighbors (k)', fontsize=10, color=C["muted"])
        ax.set_ylabel('Cross-validation accuracy', fontsize=10, color=C["muted"])
        ax.set_title('KNN accuracy by number of neighbors', fontsize=13, fontweight='bold',
                     loc="left", pad=14)

        canvas = FigureCanvasTkAgg(fig, master=self.model_plot_frame)
        canvas.draw()
        self.embed_figure(canvas)

        self.metrics["knn"].config(text=f"{best_score:.1%}", fg=C["green"])
        self.metrics["best_k"].config(text=str(best_k))
        self.log(f"\nBest k = {best_k}  →  accuracy {best_score:.4f}\n", "good")
        self.end_run(f"KNN complete — best k = {best_k}, accuracy {best_score:.1%}")

    def evaluate_rf(self):
        if not self.begin_run("Random Forest evaluation"):
            return

        randomforest_classifier = RandomForestClassifier(n_estimators=10)
        score = cross_val_score(randomforest_classifier, self.X, self.y, cv=10).mean()

        self.log("  Algorithm     Random Forest\n")
        self.log("  Estimators    10 trees\n")
        self.log("  Validation    10-fold cross-validation\n")
        self.log(f"\nAverage accuracy {score:.4f}  ({score * 100:.2f}%)\n", "good")

        self.metrics["rf"].config(text=f"{score:.1%}", fg=C["green"])
        self.end_run(f"Random Forest complete — accuracy {score:.1%}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseApp(root)
    root.mainloop()
