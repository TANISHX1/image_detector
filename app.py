import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from feature_extractor import FeatureExtractor
from model_handler import ImageClassifier
from utils import load_image_from_url, load_image_from_file, resize_for_display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Feature Extractor & Classifier")
        self.root.geometry("1200x700")
        
        # Initialize models
        self.status_var = tk.StringVar(value="Initializing models...")
        self.create_status_bar()
        self.root.update()
        
        self.feature_extractor = FeatureExtractor()
        self.classifier = ImageClassifier()
        
        self.status_var.set("Ready")
        
        # Current image
        self.current_image = None
        self.current_features = None
        
        # Create GUI
        self.create_widgets()
        
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=5, pady=2)
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Image loading and display
        left_frame = ttk.LabelFrame(main_frame, text="Image Input", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # URL input
        ttk.Label(left_frame, text="Image URL:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.url_entry = ttk.Entry(left_frame, width=40)
        self.url_entry.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.url_entry.insert(0, "https://images.unsplash.com/photo-1574158622682-e40e69881006")
        
        ttk.Button(left_frame, text="Load from URL", 
                  command=self.load_from_url).grid(row=1, column=0, pady=5)
        ttk.Button(left_frame, text="Load from File", 
                  command=self.load_from_file).grid(row=1, column=1, pady=5)
        
        # Image display
        self.image_label = ttk.Label(left_frame, text="No image loaded", 
                                     relief=tk.SUNKEN, anchor=tk.CENTER)
        self.image_label.grid(row=2, column=0, columnspan=3, pady=10, 
                             sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Action buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Extract Features", 
                  command=self.extract_features).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Classify Image", 
                  command=self.classify_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show All", 
                  command=self.show_all).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Results
        right_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Results notebook (tabs)
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Classification tab
        self.classification_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.classification_frame, text="Classification")
        
        self.classification_text = tk.Text(self.classification_frame, 
                                          width=50, height=15, wrap=tk.WORD)
        self.classification_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Features tab
        self.features_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.features_frame, text="Features")
        
        self.features_text = tk.Text(self.features_frame, 
                                    width=50, height=15, wrap=tk.WORD)
        self.features_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visualization tab
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(2, weight=1)
        
    def load_from_url(self):
        """Load image from URL"""
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Warning", "Please enter a URL")
            return
        
        try:
            self.status_var.set("Loading image from URL...")
            self.root.update()
            
            self.current_image = load_image_from_url(url)
            self.display_image(self.current_image)
            self.status_var.set("Image loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error loading image")
    
    def load_from_file(self):
        """Load image from file"""
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), 
                      ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            self.status_var.set("Loading image from file...")
            self.root.update()
            
            self.current_image = load_image_from_file(filepath)
            self.display_image(self.current_image)
            self.status_var.set("Image loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error loading image")
    
    def display_image(self, img):
        """Display image in GUI"""
        # Resize for display
        display_img = resize_for_display(img, max_size=400)
        
        # If numpy array (cv2), convert BGR->RGB
        if isinstance(display_img, np.ndarray):
            if display_img.ndim == 3 and display_img.shape[2] == 3:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(display_img)
        else:
            img_pil = display_img  # assume PIL Image
        
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Update label
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk  # Keep reference
    
    def extract_features(self):
        """Extract features from current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.status_var.set("Extracting features...")
            self.root.update()
            
            self.current_features = self.feature_extractor.extract_all_features(
                self.current_image
            )
            
            # Display features
            self.features_text.delete(1.0, tk.END)
            self.features_text.insert(tk.END, "=== EXTRACTED FEATURES ===\n\n")
            
            # Color histogram
            ch = self.current_features.get('color_histogram', [])
            self.features_text.insert(tk.END, "Color Histogram Features:\n")
            self.features_text.insert(tk.END, f"  - Dimension: {len(ch)}\n")
            self.features_text.insert(tk.END, f"  - Sample values: {ch[:5]}\n\n")
            
            # Edge features
            ef = self.current_features.get('edge_features', np.array([]))
            self.features_text.insert(tk.END, "Edge Features:\n")
            try:
                ef_len = len(ef)
            except Exception:
                ef_len = 0
            self.features_text.insert(tk.END, f"  - Dimension: {ef_len}\n")
            self.features_text.insert(tk.END, f"  - Values: {ef}\n\n")
            
            # Deep features
            df = self.current_features.get('deep_features', [])
            self.features_text.insert(tk.END, "Deep Learning Features (VGG16):\n")
            self.features_text.insert(tk.END, f"  - Dimension: {len(df)}\n")
            self.features_text.insert(tk.END, f"  - Sample values: {df[:5]}\n\n")
            
            # Switch to features tab
            self.notebook.select(self.features_frame)
            
            # Visualize features
            self.visualize_features()
            
            self.status_var.set("Features extracted successfully")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error extracting features")
    
    def classify_image(self):
        """Classify current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.status_var.set("Classifying image...")
            self.root.update()
            
            results = self.classifier.classify_image(self.current_image)
            
            # Display results
            self.classification_text.delete(1.0, tk.END)
            self.classification_text.insert(tk.END, "=== CLASSIFICATION RESULTS ===\n\n")
            self.classification_text.insert(tk.END, "Top 5 Predictions:\n")
            
            for i, result in enumerate(results, 1):
                cls = result.get('class', 'unknown')
                conf = result.get('confidence', 0.0)
                # assume confidence is percentage (0-100)
                self.classification_text.insert(tk.END, f"{i}. {cls}\n")
                self.classification_text.insert(tk.END, f"   Confidence: {conf:.2f}%\n\n")
            
            # Switch to classification tab
            self.notebook.select(self.classification_frame)
            
            self.status_var.set("Classification complete")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error classifying image")
    
    def visualize_features(self):
        """Visualize extracted features"""
        if self.current_features is None:
            return
        
        # Clear previous visualizations
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Color histogram
        color_hist = np.array(self.current_features.get('color_histogram', []))
        if color_hist.size:
            axes[0, 0].plot(color_hist)
        axes[0, 0].set_title('Color Histogram Features')
        axes[0, 0].set_xlabel('Bin')
        axes[0, 0].set_ylabel('Normalized Count')
        
        # Edge features (as heatmap)
        edge_feat = np.array(self.current_features.get('edge_features', []))
        if edge_feat.size:
            try:
                edge_grid = edge_feat.reshape(4, 4)
                im = axes[0, 1].imshow(edge_grid, cmap='hot')
                plt.colorbar(im, ax=axes[0, 1])
            except Exception:
                axes[0, 1].text(0.5, 0.5, "Edge data not 4x4", ha='center')
        axes[0, 1].set_title('Edge Density Map (4x4 Grid)')
        
        # Deep features (first 100 values)
        deep_feat = np.array(self.current_features.get('deep_features', []))
        if deep_feat.size:
            deep_plot = deep_feat[:100]
            axes[1, 0].bar(range(len(deep_plot)), deep_plot)
        axes[1, 0].set_title('Deep Features (First 100 dimensions)')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Value')
        
        # Original image
        try:
            axes[1, 1].imshow(self.current_image if isinstance(self.current_image, np.ndarray) else np.array(self.current_image))
        except Exception:
            axes[1, 1].text(0.5, 0.5, "Image not displayable", ha='center')
        axes[1, 1].set_title('Original Image')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_all(self):
        """Extract features and classify image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.extract_features()
        self.classify_image()

def main():
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()