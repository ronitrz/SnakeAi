import gzip, json, struct


class ReplayReader:
    def __init__(self, path):
        self.path = path
        self.f = None
        self.header = None
        self.width = None
        self.height = None
        self.size = None
        self.num_features = None
        self.feature_names = None
        self.cell_dtype = None
        self.board_bytes_len = None

        self._open()
        self._read_header()

    def _open(self):
        if self.path.endswith(".gz"):
            self.f = gzip.open(self.path, "rb")
        else:
            self.f = open(self.path, "rb")

    def close(self):
        if self.f:
            self.f.close()

    def _read_header(self):
        header_len = struct.unpack("<I", self.f.read(4))[0]
        self.header = json.loads(self.f.read(header_len).decode("utf-8"))

        self.width = self.header["width"]
        self.height = self.header["height"]
        self.size = self.width * self.height
        self.num_features = self.header.get("num_features", 0)
        self.feature_names = self.header.get("feature_names", None)
        self.cell_dtype = self.header["cell_dtype"]

        bytes_per_cell = 1 if self.cell_dtype == "B" else 2
        self.board_bytes_len = self.size * bytes_per_cell

    def read_record(self):
        data = self.f.read(8)
        if not data or len(data) < 8:
            return None

        episode_id, step_id = struct.unpack("<II", data)
        board = self._unpack_board(self.f.read(self.board_bytes_len))
        action, curr_dir, done = self._unpack_meta(self.f.read(1))

        features = None
        if self.num_features > 0:
            feat_bytes = self.f.read(self.num_features * 4)
            features = list(struct.unpack(f"<{self.num_features}f", feat_bytes))

        return episode_id, step_id, board, action, curr_dir, done, features

    def _unpack_board(self, board_bytes):
        cells = list(struct.unpack(f"<{self.size}{self.cell_dtype}", board_bytes))
        board = []
        for row in range(self.height):
            start = row * self.width
            board.append(cells[start : start + self.width])
        return board

    def _unpack_meta(self, meta_byte):
        b = meta_byte[0]
        return b & 3, (b >> 2) & 3, bool((b >> 4) & 1)
