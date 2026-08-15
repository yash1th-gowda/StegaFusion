"""
StegaFusion Review Demo UI

Tkinter interface for the production StegaFusion pipeline.

Provides:
    - Cover video selection
    - Secret file selection
    - AES key selection
    - Capacity analysis
    - Production embedding
    - Production extraction
    - SHA-256 verification
    - Byte-for-byte verification
    - PSNR / SSIM quality display for the
      configured demonstration video

The UI does not implement steganography.
It calls the existing production APIs.
"""

from pathlib import Path
import hashlib
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


from modules.pipeline.spatial_video_pipeline import (
    embed_spatial_video,
    extract_spatial_video,
)

from modules.evaluation.capacity import (
    analyze_capacity,
)


# ==========================================================
# DEFAULT CONFIGURATION
# ==========================================================

DEFAULT_COVER = Path(
    "input/cover_video/sample_long.mp4"
)

DEFAULT_SECRET = Path(
    "input/secret_data/scalability_test/secret_32kb.bin"
)

DEFAULT_KEY = Path(
    "input/keys/test_aes_key.bin"
)

DEFAULT_OUTPUT_DIR = Path(
    "output/ui_demo"
)

DELTA = 5


# ==========================================================
# SHA-256
# ==========================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ==========================================================
# APPLICATION
# ==========================================================

class StegaFusionApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "StegaFusion — Secure Video Steganography"
        )

        self.root.geometry(
            "1050x760"
        )

        self.root.minsize(
            900,
            650
        )

        self._build_ui()

    # ======================================================
    # UI
    # ======================================================

    def _build_ui(self):

        main = ttk.Frame(
            self.root,
            padding=20
        )

        main.pack(
            fill="both",
            expand=True
        )

        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------

        title = ttk.Label(
            main,
            text="STEGAFUSION",
            font=(
                "Segoe UI",
                24,
                "bold"
            )
        )

        title.pack(
            anchor="w"
        )

        subtitle = ttk.Label(
            main,
            text=(
                "Secure Video Steganography "
                "and Experimental Evaluation"
            ),
            font=(
                "Segoe UI",
                11
            )
        )

        subtitle.pack(
            anchor="w",
            pady=(0, 20)
        )

        # --------------------------------------------------
        # INPUT SECTION
        # --------------------------------------------------

        input_frame = ttk.LabelFrame(
            main,
            text="Input Configuration",
            padding=15
        )

        input_frame.pack(
            fill="x"
        )

        self.cover_var = tk.StringVar(
            value=str(DEFAULT_COVER)
        )

        self.secret_var = tk.StringVar(
            value=str(DEFAULT_SECRET)
        )

        self.key_var = tk.StringVar(
            value=str(DEFAULT_KEY)
        )

        self._create_file_row(
            input_frame,
            "Cover Video",
            self.cover_var,
            self.browse_video
        )

        self._create_file_row(
            input_frame,
            "Secret File",
            self.secret_var,
            self.browse_secret
        )

        self._create_file_row(
            input_frame,
            "AES Key",
            self.key_var,
            self.browse_key
        )

        # --------------------------------------------------
        # CONTROLS
        # --------------------------------------------------

        controls = ttk.Frame(
            main
        )

        controls.pack(
            fill="x",
            pady=15
        )

        self.capacity_button = ttk.Button(
            controls,
            text="Analyze Capacity",
            command=self.run_capacity
        )

        self.capacity_button.pack(
            side="left",
            padx=(0, 8)
        )

        self.embed_button = ttk.Button(
            controls,
            text="Embed Secret",
            command=self.run_embedding
        )

        self.embed_button.pack(
            side="left",
            padx=8
        )

        self.extract_button = ttk.Button(
            controls,
            text="Extract Secret",
            command=self.run_extraction
        )

        self.extract_button.pack(
            side="left",
            padx=8
        )

        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------

        status_frame = ttk.LabelFrame(
            main,
            text="System Status",
            padding=15
        )

        status_frame.pack(
            fill="x",
            pady=(0, 15)
        )

        self.status_var = tk.StringVar(
            value="Ready."
        )

        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=(
                "Segoe UI",
                11,
                "bold"
            )
        )

        self.status_label.pack(
            anchor="w"
        )

        self.progress = ttk.Progressbar(
            status_frame,
            mode="indeterminate"
        )

        self.progress.pack(
            fill="x",
            pady=(10, 0)
        )

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        results_frame = ttk.LabelFrame(
            main,
            text="Experimental Results",
            padding=10
        )

        results_frame.pack(
            fill="both",
            expand=True
        )

        self.output = tk.Text(
            results_frame,
            wrap="word",
            font=(
                "Consolas",
                10
            ),
            state="disabled"
        )

        scrollbar = ttk.Scrollbar(
            results_frame,
            orient="vertical",
            command=self.output.yview
        )

        self.output.configure(
            yscrollcommand=scrollbar.set
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.output.pack(
            side="left",
            fill="both",
            expand=True
        )

        # --------------------------------------------------
        # INITIAL MESSAGE
        # --------------------------------------------------

        self.write_output(
            "StegaFusion review demonstration ready.\n\n"
            "Recommended demonstration:\n"
            "1. Analyze capacity\n"
            "2. Embed the secret\n"
            "3. Extract the secret\n"
            "4. Verify SHA-256 and byte-for-byte integrity\n"
        )

    # ======================================================
    # FILE ROW
    # ======================================================

    def _create_file_row(
        self,
        parent,
        label,
        variable,
        browse_command
    ):

        row = ttk.Frame(
            parent
        )

        row.pack(
            fill="x",
            pady=5
        )

        ttk.Label(
            row,
            text=label,
            width=14
        ).pack(
            side="left"
        )

        ttk.Entry(
            row,
            textvariable=variable
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=8
        )

        ttk.Button(
            row,
            text="Browse",
            command=browse_command
        ).pack(
            side="right"
        )

    # ======================================================
    # BROWSERS
    # ======================================================

    def browse_video(self):

        path = filedialog.askopenfilename(
            title="Select Cover Video",
            filetypes=[
                (
                    "Video files",
                    "*.mp4 *.avi *.mov *.mkv"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if path:
            self.cover_var.set(path)

    def browse_secret(self):

        path = filedialog.askopenfilename(
            title="Select Secret File",
            filetypes=[
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if path:
            self.secret_var.set(path)

    def browse_key(self):

        path = filedialog.askopenfilename(
            title="Select AES Key",
            filetypes=[
                (
                    "Key files",
                    "*.bin *.key"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if path:
            self.key_var.set(path)

    # ======================================================
    # OUTPUT
    # ======================================================

    def write_output(
        self,
        text
    ):

        self.output.configure(
            state="normal"
        )

        self.output.insert(
            "end",
            text
        )

        self.output.see(
            "end"
        )

        self.output.configure(
            state="disabled"
        )

    def clear_output(self):

        self.output.configure(
            state="normal"
        )

        self.output.delete(
            "1.0",
            "end"
        )

        self.output.configure(
            state="disabled"
        )

    # ======================================================
    # STATUS
    # ======================================================

    def set_busy(
        self,
        message
    ):

        self.status_var.set(
            message
        )

        self.progress.start(
            10
        )

        self.capacity_button.configure(
            state="disabled"
        )

        self.embed_button.configure(
            state="disabled"
        )

        self.extract_button.configure(
            state="disabled"
        )

    def set_ready(
        self,
        message="Ready."
    ):

        self.status_var.set(
            message
        )

        self.progress.stop()

        self.capacity_button.configure(
            state="normal"
        )

        self.embed_button.configure(
            state="normal"
        )

        self.extract_button.configure(
            state="normal"
        )

    # ======================================================
    # CAPACITY
    # ======================================================

    def run_capacity(self):

        self.set_busy(
            "Analyzing video capacity..."
        )

        self.write_output(
            "\n"
            + "=" * 70
            + "\n"
            + "CAPACITY ANALYSIS\n"
            + "=" * 70
            + "\n"
        )

        thread = threading.Thread(
            target=self._capacity_worker,
            daemon=True
        )

        thread.start()

    def _capacity_worker(self):

        try:

            report = analyze_capacity(
                video=Path(
                    self.cover_var.get()
                ),
                secret_file=Path(
                    self.secret_var.get()
                ),
                key_file=Path(
                    self.key_var.get()
                ),
                delta=DELTA,
            )

            text = (
                f"Secret Size       : "
                f"{report.secret_bytes} bytes\n"
                f"Payload Bits      : "
                f"{report.payload_bits}\n"
                f"Packet Size       : "
                f"{report.packet_bits} bits\n"
                f"Packet Count      : "
                f"{report.packet_count}\n"
                f"Spatial Capacity  : "
                f"{report.spatial_capacity_bits} bits/frame\n"
                f"Required Frames   : "
                f"{report.required_frames}\n"
                f"Available Frames  : "
                f"{report.available_frames}\n"
                f"Remaining Frames  : "
                f"{report.remaining_frames}\n"
                f"Delta             : "
                f"{DELTA}\n"
                f"\n"
                f"CAPACITY STATUS   : "
                f"{'PASS' if report.fits_video else 'FAIL'}\n\n"
            )

            self.root.after(
                0,
                lambda: self.write_output(
                    text
                )
            )

            self.root.after(
                0,
                lambda: self.set_ready(
                    "Capacity analysis complete."
                )
            )

        except Exception as exc:

            self.root.after(
                0,
                lambda: self.write_output(
                    f"\nERROR: {exc}\n"
                )
            )

            self.root.after(
                0,
                lambda: self.set_ready(
                    "Capacity analysis failed."
                )
            )

    # ======================================================
    # EMBEDDING
    # ======================================================

    def run_embedding(self):

        self.set_busy(
            "Embedding secret..."
        )

        self.write_output(
            "\n"
            + "=" * 70
            + "\n"
            + "PRODUCTION EMBEDDING\n"
            + "=" * 70
            + "\n"
        )

        thread = threading.Thread(
            target=self._embedding_worker,
            daemon=True
        )

        thread.start()

    def _embedding_worker(self):

        try:

            output_dir = (
                DEFAULT_OUTPUT_DIR
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_video = (
                output_dir
                / "stego_demo.mp4"
            )

            result = embed_spatial_video(
                cover_video=Path(
                    self.cover_var.get()
                ),
                secret_file=Path(
                    self.secret_var.get()
                ),
                key_file=Path(
                    self.key_var.get()
                ),
                output_video=output_video,
                delta=DELTA,
            )

            text = (
                "Embedding : PASS\n\n"
                f"Payload Bits      : "
                f"{result['payload_bits']}\n"
                f"Embedded Bits     : "
                f"{result['embedded_bits']}\n"
                f"Packet Count      : "
                f"{result['packet_count']}\n"
                f"Embedded Frames   : "
                f"{result['embedded_frames']}\n"
                f"Video Frames      : "
                f"{result['frames']}\n"
                f"Delta             : "
                f"{DELTA}\n"
                f"Output Video      : "
                f"{result['output_video']}\n\n"
                "STEGO VIDEO CREATED : YES\n\n"
            )

            self.root.after(
                0,
                lambda: self.write_output(
                    text
                )
            )

            self.root.after(
                0,
                lambda: self.set_ready(
                    "Embedding complete."
                )
            )

        except Exception as exc:

            self.root.after(
                0,
                lambda: self.write_output(
                    f"\nERROR: {exc}\n"
                )
            )

            self.root.after(
                0,
                lambda: self.set_ready(
                    "Embedding failed."
                )
            )

    # ======================================================
    # EXTRACTION
    # ======================================================

    def run_extraction(self):

        self.set_busy(
            "Extracting secret..."
        )

        self.write_output(
            "\n"
            + "=" * 70
            + "\n"
            + "PRODUCTION EXTRACTION\n"
            + "=" * 70
            + "\n"
        )

        thread = threading.Thread(
            target=self._extraction_worker,
            daemon=True
        )

        thread.start()

    def _extraction_worker(self):

        try:

            output_dir = (
                DEFAULT_OUTPUT_DIR
                / "recovered"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            stego_video = (
                DEFAULT_OUTPUT_DIR
                / "stego_demo.mp4"
            )

            recovered_file = (
                output_dir
                / "recovered_secret.bin"
            )

            if not stego_video.exists():

                raise FileNotFoundError(
                    "No UI-generated stego video exists.\n"
                    "Run Embed Secret first."
                )

            result = extract_spatial_video(
                cover_video=Path(
                    self.cover_var.get()
                ),
                stego_video=stego_video,
                key_file=Path(
                    self.key_var.get()
                ),
                output_file=recovered_file,
                delta=DELTA,
            )

            original_file = Path(
                self.secret_var.get()
            )

            original_bytes = (
                original_file.read_bytes()
            )

            recovered_bytes = (
                recovered_file.read_bytes()
            )

            byte_match = (
                original_bytes
                == recovered_bytes
            )

            original_hash = (
                sha256_file(
                    original_file
                )
            )

            recovered_hash = (
                sha256_file(
                    recovered_file
                )
            )

            hash_match = (
                original_hash
                == recovered_hash
            )

            text = (
                "Extraction : PASS\n\n"
                f"Recovered Bytes : "
                f"{result['recovered_bytes']}\n"
                f"Recovered File  : "
                f"{recovered_file}\n\n"
                "BYTE-FOR-BYTE VALIDATION\n"
                f"Match : "
                f"{'TRUE' if byte_match else 'FALSE'}\n\n"
                "SHA-256 VALIDATION\n"
                f"Original  : "
                f"{original_hash}\n"
                f"Recovered : "
                f"{recovered_hash}\n"
                f"Match     : "
                f"{'TRUE' if hash_match else 'FALSE'}\n\n"
            )

            if (
                byte_match
                and hash_match
            ):

                text += (
                    "=" * 50
                    + "\n"
                    "INTEGRITY VERIFIED\n"
                    + "=" * 50
                    + "\n"
                )

            else:

                text += (
                    "INTEGRITY VERIFICATION FAILED\n"
                )

            self.root.after(
                0,
                lambda: self.write_output(
                    text
                )
            )

            self.root.after(
                0,
                lambda: self.set_ready(
                    "Extraction and verification complete."
                )
            )

        except Exception as exc:

            self.root.after(
                0,
                lambda: self.write_output(
                    f"\nERROR: {exc}\n"
                )
            )

            self.root.after(
                0,
                lambda: self.set_ready(
                    "Extraction failed."
                )
            )


# ==========================================================
# ENTRY POINT
# ==========================================================

def main():

    root = tk.Tk()

    app = StegaFusionApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()