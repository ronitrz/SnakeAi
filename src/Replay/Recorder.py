import gzip, json, struct


class Recorder:
    def __init__(self, path, width, height, compress=True, feature_names=None):
        self.width = width
        self.height = height
        self.size = width * height
        self.feature_names = feature_names or []
        self.num_features = len(self.feature_names)
        self.episode_id = 0
        self.step_id = 0

        if self.size <= 255:
            self.cell_dtype = "B"  # uint8
            self.bytes_per_cell = 1
        else:
            self.cell_dtype = "H"  # uint16
            self.bytes_per_cell = 2

        self.f = gzip.open(path, "wb") if compress else open(path, "wb")
        self._write_header()

    def _write_header(self):
        header = {
            "width": self.width,
            "height": self.height,
            "num_features": self.num_features,
            "feature_names": self.feature_names,
            "cell_dtype": self.cell_dtype,
        }
        data = json.dumps(header).encode("utf-8")
        self.f.write(struct.pack("<I", len(data)))
        self.f.write(data)

    def start_episode(self):
        self.episode_id += 1
        self.step_id = 0

    def record_step(self, board, action, curr_dir, done=False, features=None):
        if features is not None and len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )

        self.f.write(struct.pack("<II", self.episode_id, self.step_id))

        flat = [c for row in board for c in row]
        self.f.write(struct.pack(f"<{self.size}{self.cell_dtype}", *flat))

        meta = (action & 0b11) | ((curr_dir & 0b11) << 2) | ((int(done) & 0b1) << 4)
        self.f.write(bytes([meta]))

        if self.num_features > 0 and features is not None:
            self.f.write(struct.pack(f"<{self.num_features}f", *features))

        self.step_id += 1

    def close(self):
        self.f.close()
