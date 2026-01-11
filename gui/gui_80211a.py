import os
import sys
import numpy as np

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QBrush, QPen, QFont, QPainterPath, QPainter, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPathItem,
    QGraphicsTextItem,
    QDialog,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QTabWidget,
    QSizePolicy,
    QSplitter,
    QMessageBox,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


def _safe_import():
    try:
        from tx.OFDM_TX_802_11 import Transmitter80211a
        from channel.Channel_Model import Channel_Model
        from channel.channel_settings import ChannelSettings
        from channel.channel_mode import ChannelMode
        from rx.RX_802_11_a import Receiver80211a
        return {
            "Transmitter80211a": Transmitter80211a,
            "Channel_Model": Channel_Model,
            "ChannelSettings": ChannelSettings,
            "ChannelMode": ChannelMode,
            "Receiver80211a": Receiver80211a,
        }
    except Exception as e:
        return {"error": str(e)}


def _bps_from_mod(name: str) -> int:
    if name == "BPSK":
        return 1
    if name == "QPSK":
        return 2
    if name == "16-QAM":
        return 4
    return 2


def _fft_spectrum_db(x, fs):
    x = np.asarray(x).flatten()
    n = len(x)
    if n <= 1:
        return np.zeros(0), np.zeros(0)
    w = np.hanning(n)
    X = np.fft.fftshift(np.fft.fft(x * w))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    mag = 20 * np.log10(np.maximum(np.abs(X), 1e-12))
    mag -= np.max(mag)
    return f, mag


def _plot_td(ax, sig, fs, title):
    sig = np.asarray(sig).flatten()
    nshow = min(len(sig), 5000)
    t = np.arange(nshow) / fs
    ax.plot(t * 1e6, np.real(sig[:nshow]))
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("Vrijeme [µs]")
    ax.set_ylabel("Amplituda")


def _plot_spec(ax, sig, fs, title):
    f, mag = _fft_spectrum_db(sig, fs)
    ax.plot(f / 1e6, mag)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("Frekvencija [MHz]")
    ax.set_ylabel("dB")


def _plot_const(ax, syms, title):
    syms = None if syms is None else np.asarray(syms).flatten()
    if syms is None or len(syms) == 0:
        ax.text(0.5, 0.5, "Nema simbola", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.scatter(syms.real, syms.imag, s=7, alpha=0.85)
    ax.grid(True)
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")


def _extract_constellation_from_td(rx_td, up, num_syms):
    rx_td = np.asarray(rx_td).flatten()
    sts = 16 * up * 10
    lts = 160 * up
    cp = 16 * up
    n = 64 * up
    sym = cp + n
    start_payload = sts + lts
    out = []
    for i in range(int(num_syms)):
        s0 = start_payload + i * sym
        s1 = s0 + cp
        s2 = s1 + n
        if s2 > len(rx_td):
            break
        sym_td = rx_td[s1:s2]
        if up > 1:
            sym_td = sym_td[::up]
            N = 64
        else:
            N = 64
        if len(sym_td) != N:
            continue
        sym_fd = (1 / N) * np.fft.fft(sym_td)
        out.append(sym_fd)
    if len(out) == 0:
        return None
    return np.concatenate(out)


class MplWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.setLayout(lay)

    def ax(self):
        self.fig.clf()
        return self.fig.add_subplot(111)

    def draw(self):
        self.fig.tight_layout()
        self.canvas.draw()


class RoundedBlockItem(QGraphicsPathItem):
    def __init__(self, title, key, rect: QRectF, fill: QColor, on_open):
        super().__init__()
        self.key = key
        self.on_open = on_open
        r = 18.0
        path = QPainterPath()
        path.addRoundedRect(rect, r, r)
        self.setPath(path)
        self.setBrush(QBrush(fill))
        self.setPen(QPen(QColor("#1f2937"), 2))
        self.setFlag(QGraphicsPathItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        self.text = QGraphicsTextItem(title, self)
        f = QFont()
        f.setPointSize(14)
        f.setBold(True)
        self.text.setFont(f)
        self.text.setDefaultTextColor(QColor("#111827"))
        tr = self.text.boundingRect()
        self.text.setPos(rect.x() + (rect.width() - tr.width()) / 2, rect.y() + (rect.height() - tr.height()) / 2)

        self._rect = rect
        self._base = fill
        self._hover = QColor(fill)
        self._hover.setAlpha(230)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(self._hover))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(self._base))
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if callable(self.on_open):
            self.on_open(self.key)
        super().mouseDoubleClickEvent(event)


class WireItem(QGraphicsPathItem):
    def __init__(self, p1: QPointF, p2: QPointF):
        super().__init__()
        self.setPen(QPen(QColor("#111827"), 2))
        path = QPainterPath()
        path.moveTo(p1)
        dx = (p2.x() - p1.x()) * 0.5
        c1 = QPointF(p1.x() + dx, p1.y())
        c2 = QPointF(p2.x() - dx, p2.y())
        path.cubicTo(c1, c2, p2)
        self.setPath(path)


class SetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Podešavanje simulacije")

        self.mod = QComboBox()
        self.mod.addItems(["BPSK", "QPSK", "16-QAM"])
        self.mod.setCurrentText("QPSK")

        self.up_factor = QComboBox()
        self.up_factor.addItems(["1", "2", "4"])
        self.up_factor.setCurrentText("2")

        self.payload_samples = QSpinBox()
        self.payload_samples.setRange(80, 10_000_000)
        self.payload_samples.setValue(200_000)

        self.snr_db = QDoubleSpinBox()
        self.snr_db.setRange(-5.0, 60.0)
        self.snr_db.setDecimals(1)
        self.snr_db.setValue(20.0)

        self.multipath = QCheckBox("Multipath")
        self.multipath.setChecked(True)

        self.num_taps = QSpinBox()
        self.num_taps.setRange(1, 30)
        self.num_taps.setValue(2)

        self.delay_spread = QDoubleSpinBox()
        self.delay_spread.setRange(0.0, 5e-6)
        self.delay_spread.setDecimals(9)
        self.delay_spread.setValue(10e-9)

        self.btn_ok = QPushButton("Primijeni")
        self.btn_ok.clicked.connect(self.accept)

        self.btn_cancel = QPushButton("Otkaži")
        self.btn_cancel.clicked.connect(self.reject)

        form = QFormLayout()
        form.addRow("Modulacija", self.mod)
        form.addRow("Upsampling faktor", self.up_factor)
        form.addRow("Broj uzoraka (payload)", self.payload_samples)
        form.addRow("SNR [dB]", self.snr_db)
        form.addRow(self.multipath)
        form.addRow("Broj tapova", self.num_taps)
        form.addRow("Delay spread [s]", self.delay_spread)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_ok)

        lay = QVBoxLayout()
        lay.addLayout(form)
        lay.addLayout(row)
        self.setLayout(lay)
        self.resize(420, 320)

    def get_config(self):
        return {
            "mod": self.mod.currentText(),
            "bps": _bps_from_mod(self.mod.currentText()),
            "up": int(self.up_factor.currentText()),
            "payload_samples": int(self.payload_samples.value()),
            "snr_db": float(self.snr_db.value()),
            "multipath": bool(self.multipath.isChecked()),
            "num_taps": int(self.num_taps.value()),
            "delay_spread": float(self.delay_spread.value()),
        }


class CompareDialog(QDialog):
    def __init__(self, parent=None, title="Uporedni prikaz"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout()
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(self.canvas)
        self.setLayout(lay)
        self.resize(1200, 820)

    def plot_3x3(self, tx_sig, ch_sig, rx_sig, fs, tx_const=None, ch_const=None, rx_const=None):
        self.fig.clf()
        axs = [self.fig.add_subplot(3, 3, i + 1) for i in range(9)]

        tx_sig = np.asarray(tx_sig).flatten()
        ch_sig = np.asarray(ch_sig).flatten()
        rx_sig = np.asarray(rx_sig).flatten()

        nshow = min(len(tx_sig), len(ch_sig), len(rx_sig), 5000)

        t = np.arange(nshow) / fs

        axs[0].plot(t * 1e6, np.real(tx_sig[:nshow]))
        axs[0].grid(True)
        axs[0].set_title("TX: vrijeme")
        axs[0].set_xlabel("µs")

        f_tx, m_tx = _fft_spectrum_db(tx_sig, fs)
        axs[1].plot(f_tx / 1e6, m_tx)
        axs[1].grid(True)
        axs[1].set_title("TX: spektar")
        axs[1].set_xlabel("MHz")

        if tx_const is not None and len(np.asarray(tx_const).flatten()) > 0:
            c = np.asarray(tx_const).flatten()
            axs[2].scatter(c.real, c.imag, s=6)
            axs[2].grid(True)
            axs[2].set_aspect("equal", "box")
        axs[2].set_title("TX: konstelacija")

        axs[3].plot(t * 1e6, np.real(ch_sig[:nshow]))
        axs[3].grid(True)
        axs[3].set_title("Kanal: vrijeme")
        axs[3].set_xlabel("µs")

        f_ch, m_ch = _fft_spectrum_db(ch_sig, fs)
        axs[4].plot(f_ch / 1e6, m_ch)
        axs[4].grid(True)
        axs[4].set_title("Kanal: spektar")
        axs[4].set_xlabel("MHz")

        if ch_const is not None and len(np.asarray(ch_const).flatten()) > 0:
            c = np.asarray(ch_const).flatten()
            axs[5].scatter(c.real, c.imag, s=6)
            axs[5].grid(True)
            axs[5].set_aspect("equal", "box")
        axs[5].set_title("Kanal: konstelacija")

        axs[6].plot(t * 1e6, np.real(rx_sig[:nshow]))
        axs[6].grid(True)
        axs[6].set_title("RX: vrijeme")
        axs[6].set_xlabel("µs")

        f_rx, m_rx = _fft_spectrum_db(rx_sig, fs)
        axs[7].plot(f_rx / 1e6, m_rx)
        axs[7].grid(True)
        axs[7].set_title("RX: spektar")
        axs[7].set_xlabel("MHz")

        if rx_const is not None and len(np.asarray(rx_const).flatten()) > 0:
            c = np.asarray(rx_const).flatten()
            axs[8].scatter(c.real, c.imag, s=6)
            axs[8].grid(True)
            axs[8].set_aspect("equal", "box")
        axs[8].set_title("RX: konstelacija")

        self.fig.tight_layout()
        self.canvas.draw()


class TxDialog(QDialog):
    def __init__(self, parent, modules, sim):
        super().__init__(parent)
        self.modules = modules
        self.sim = sim
        self.setWindowTitle("TX (802.11a)")

        self.plot_td = MplWidget()
        self.plot_spec = MplWidget()
        self.plot_const = MplWidget()

        self.info = QLabel("")
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.btn_gen = QPushButton("Generiši TX")
        self.btn_gen.clicked.connect(self.run)

        self.btn_cmp = QPushButton("Uporedi (3×3)")
        self.btn_cmp.clicked.connect(self.compare)

        left = QWidget()
        form = QFormLayout()
        self.lbl_mod = QLabel("")
        self.lbl_up = QLabel("")
        self.lbl_snr = QLabel("")
        form.addRow(self.lbl_mod)
        form.addRow(self.lbl_up)
        form.addRow(self.lbl_snr)
        form.addRow(self.btn_gen)
        form.addRow(self.btn_cmp)
        form.addRow(self.info)
        left.setLayout(form)

        tabs = QTabWidget()
        tabs.addTab(self.plot_td, "Vrijeme")
        tabs.addTab(self.plot_spec, "Spektar")
        tabs.addTab(self.plot_const, "Konstelacija")

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        lay = QVBoxLayout()
        lay.addWidget(splitter)
        self.setLayout(lay)
        self.resize(1200, 700)

        self._refresh_labels()

    def _refresh_labels(self):
        self.lbl_mod.setText(f"Modulacija: {self.sim['mod']}")
        self.lbl_up.setText(f"Upsampling: {self.sim['up']}")
        self.lbl_snr.setText(f"SNR: {self.sim['snr_db']:.1f} dB")

    def run(self):
        self._refresh_labels()

        if "error" in self.modules:
            QMessageBox.critical(self, "Import error", self.modules["error"])
            return

        Transmitter80211a = self.modules["Transmitter80211a"]

        fs_base = 20e6
        up = int(self.sim["up"])
        fs = fs_base * up
        bps = int(self.sim["bps"])
        seed = int(np.random.randint(0, 10_000_000))

        payload_samples = int(self.sim["payload_samples"])
        sym_len = 80 * up
        num_syms = max(1, int(round(payload_samples / sym_len)))

        try:
            tx = Transmitter80211a(
                num_ofdm_symbols=num_syms,
                bits_per_symbol=bps,
                step=1,
                up_factor=up,
                seed=seed,
                plot=False,
            )
            out = tx.generate_frame()
            tx_sig = np.asarray(out[0]).flatten()

            tx_symbols = None
            if len(out) >= 3 and out[1] is not None:
                tx_symbols = out[1]
            elif len(out) >= 3 and out[2] is not None:
                tx_symbols = out[2]
            elif len(out) >= 2 and out[1] is not None:
                tx_symbols = out[1]

            if tx_symbols is not None:
                tx_symbols = np.asarray(tx_symbols).flatten()
        except Exception as e:
            QMessageBox.critical(self, "TX error", str(e))
            return

        ax = self.plot_td.ax()
        _plot_td(ax, tx_sig, fs, "TX signal u vremenu")
        self.plot_td.draw()

        ax2 = self.plot_spec.ax()
        _plot_spec(ax2, tx_sig, fs, "TX spektar (normirano)")
        self.plot_spec.draw()

        ax3 = self.plot_const.ax()
        _plot_const(ax3, tx_symbols, "TX konstelacija (payload)")
        self.plot_const.draw()

        self.sim["seed_last"] = seed
        self.sim["num_syms_last"] = num_syms
        self.sim["fs_last"] = fs
        self.sim["tx_sig"] = tx_sig
        self.sim["tx_const"] = tx_symbols

        self.info.setText(f"Seed: {seed}\nOFDM simboli (interno): {num_syms}")

    def compare(self):
        if "tx_sig" not in self.sim or self.sim["tx_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo generiši TX.")
            return
        if "ch_sig" not in self.sim or self.sim["ch_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo primijeni Kanal (ili pokreni RX).")
            return
        if "rx_sig" not in self.sim or self.sim["rx_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo pokreni RX.")
            return

        dlg = CompareDialog(self, "Uporedni prikaz (TX / Kanal / RX)")
        dlg.plot_3x3(
            self.sim["tx_sig"],
            self.sim["ch_sig"],
            self.sim["rx_sig"],
            float(self.sim["fs_last"]),
            tx_const=self.sim.get("tx_const", None),
            ch_const=self.sim.get("ch_const", None),
            rx_const=self.sim.get("rx_const", None),
        )
        dlg.exec_()


class ChannelDialog(QDialog):
    def __init__(self, parent, modules, sim):
        super().__init__(parent)
        self.modules = modules
        self.sim = sim
        self.setWindowTitle("Kanal")

        self.plot_td = MplWidget()
        self.plot_spec = MplWidget()
        self.plot_const = MplWidget()

        self.info = QLabel("")
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.btn_run = QPushButton("Primijeni kanal")
        self.btn_run.clicked.connect(self.run)

        self.btn_cmp = QPushButton("Uporedi (3×3)")
        self.btn_cmp.clicked.connect(self.compare)

        left = QWidget()
        form = QFormLayout()
        self.lbl_mod = QLabel("")
        self.lbl_up = QLabel("")
        self.lbl_snr = QLabel("")
        self.lbl_mp = QLabel("")
        form.addRow(self.lbl_mod)
        form.addRow(self.lbl_up)
        form.addRow(self.lbl_snr)
        form.addRow(self.lbl_mp)
        form.addRow(self.btn_run)
        form.addRow(self.btn_cmp)
        form.addRow(self.info)
        left.setLayout(form)

        tabs = QTabWidget()
        tabs.addTab(self.plot_td, "Vrijeme")
        tabs.addTab(self.plot_spec, "Spektar")
        tabs.addTab(self.plot_const, "Konstelacija")

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        lay = QVBoxLayout()
        lay.addWidget(splitter)
        self.setLayout(lay)
        self.resize(1200, 700)

        self._refresh_labels()

    def _refresh_labels(self):
        self.lbl_mod.setText(f"Modulacija: {self.sim['mod']}")
        self.lbl_up.setText(f"Upsampling: {self.sim['up']}")
        self.lbl_snr.setText(f"SNR: {self.sim['snr_db']:.1f} dB")
        self.lbl_mp.setText(f"Multipath: {'DA' if self.sim['multipath'] else 'NE'}")

    def run(self):
        self._refresh_labels()

        if "error" in self.modules:
            QMessageBox.critical(self, "Import error", self.modules["error"])
            return

        if "tx_sig" not in self.sim or self.sim["tx_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo generiši TX.")
            return

        Channel_Model = self.modules["Channel_Model"]
        ChannelSettings = self.modules["ChannelSettings"]
        ChannelMode = self.modules["ChannelMode"]

        tx_sig = np.asarray(self.sim["tx_sig"]).flatten()
        fs = float(self.sim["fs_last"])

        settings = ChannelSettings(
            sample_rate=fs,
            number_of_taps=int(self.sim["num_taps"]),
            delay_spread=float(self.sim["delay_spread"]),
            snr_db=float(self.sim["snr_db"]),
        )
        mode = ChannelMode(multipath=1 if self.sim["multipath"] else 0, thermal_noise=1)

        try:
            ch = Channel_Model(settings, mode)
            ch_sig, _ = ch.apply(tx_sig)
            ch_sig = np.asarray(ch_sig).flatten()
            if np.mean(np.abs(ch_sig) ** 2) > 0:
                ch_sig *= np.sqrt(np.mean(np.abs(tx_sig) ** 2)) / np.sqrt(np.mean(np.abs(ch_sig) ** 2))
        except Exception as e:
            QMessageBox.critical(self, "Channel error", str(e))
            return

        ax = self.plot_td.ax()
        nshow = min(len(tx_sig), len(ch_sig), 5000)
        t = np.arange(nshow) / fs
        ax.plot(t * 1e6, np.real(tx_sig[:nshow]), label="TX")
        ax.plot(t * 1e6, np.real(ch_sig[:nshow]), label="Kanal", alpha=0.85)
        ax.grid(True)
        ax.set_title("TX vs poslije kanala (vrijeme)")
        ax.set_xlabel("Vrijeme [µs]")
        ax.set_ylabel("Amplituda")
        ax.legend()
        self.plot_td.draw()

        ax2 = self.plot_spec.ax()
        f1, m1 = _fft_spectrum_db(tx_sig, fs)
        f2, m2 = _fft_spectrum_db(ch_sig, fs)
        ax2.plot(f1 / 1e6, m1, label="TX")
        ax2.plot(f2 / 1e6, m2, label="Kanal", alpha=0.9)
        ax2.grid(True)
        ax2.set_title("Spektar (normirano)")
        ax2.set_xlabel("Frekvencija [MHz]")
        ax2.set_ylabel("dB")
        ax2.legend()
        self.plot_spec.draw()

        up = int(self.sim["up"])
        num_syms = int(self.sim.get("num_syms_last", 1))
        ch_const = _extract_constellation_from_td(ch_sig, up, num_syms)

        ax3 = self.plot_const.ax()
        _plot_const(ax3, ch_const, "Kanal konstelacija (FFT bez EQ)")
        self.plot_const.draw()

        self.sim["ch_sig"] = ch_sig
        self.sim["ch_const"] = ch_const
        self.info.setText("Kanal primijenjen.")

    def compare(self):
        if "tx_sig" not in self.sim or self.sim["tx_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo generiši TX.")
            return
        if "ch_sig" not in self.sim or self.sim["ch_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo primijeni Kanal.")
            return
        if "rx_sig" not in self.sim or self.sim["rx_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo pokreni RX.")
            return

        dlg = CompareDialog(self, "Uporedni prikaz (TX / Kanal / RX)")
        dlg.plot_3x3(
            self.sim["tx_sig"],
            self.sim["ch_sig"],
            self.sim["rx_sig"],
            float(self.sim["fs_last"]),
            tx_const=self.sim.get("tx_const", None),
            ch_const=self.sim.get("ch_const", None),
            rx_const=self.sim.get("rx_const", None),
        )
        dlg.exec_()


class RxDialog(QDialog):
    def __init__(self, parent, modules, sim):
        super().__init__(parent)
        self.modules = modules
        self.sim = sim
        self.setWindowTitle("RX")

        self.plot_td = MplWidget()
        self.plot_spec = MplWidget()
        self.plot_const = MplWidget()

        self.info = QLabel("")
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.btn_run = QPushButton("Pokreni RX")
        self.btn_run.clicked.connect(self.run)

        self.btn_cmp = QPushButton("Uporedi (3×3)")
        self.btn_cmp.clicked.connect(self.compare)

        left = QWidget()
        form = QFormLayout()
        self.lbl_mod = QLabel("")
        self.lbl_up = QLabel("")
        self.lbl_snr = QLabel("")
        self.lbl_mp = QLabel("")
        form.addRow(self.lbl_mod)
        form.addRow(self.lbl_up)
        form.addRow(self.lbl_snr)
        form.addRow(self.lbl_mp)
        form.addRow(self.btn_run)
        form.addRow(self.btn_cmp)
        form.addRow(self.info)
        left.setLayout(form)

        tabs = QTabWidget()
        tabs.addTab(self.plot_td, "Vrijeme")
        tabs.addTab(self.plot_spec, "Spektar")
        tabs.addTab(self.plot_const, "Konstelacija")

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        lay = QVBoxLayout()
        lay.addWidget(splitter)
        self.setLayout(lay)
        self.resize(1200, 700)

        self._refresh_labels()

    def _refresh_labels(self):
        self.lbl_mod.setText(f"Modulacija: {self.sim['mod']}")
        self.lbl_up.setText(f"Upsampling: {self.sim['up']}")
        self.lbl_snr.setText(f"SNR: {self.sim['snr_db']:.1f} dB")
        self.lbl_mp.setText(f"Multipath: {'DA' if self.sim['multipath'] else 'NE'}")

    def run(self):
        self._refresh_labels()

        if "error" in self.modules:
            QMessageBox.critical(self, "Import error", self.modules["error"])
            return

        if "tx_sig" not in self.sim or self.sim["tx_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo generiši TX.")
            return

        Channel_Model = self.modules["Channel_Model"]
        ChannelSettings = self.modules["ChannelSettings"]
        ChannelMode = self.modules["ChannelMode"]
        Receiver80211a = self.modules["Receiver80211a"]

        tx_sig = np.asarray(self.sim["tx_sig"]).flatten()
        fs = float(self.sim["fs_last"])
        up = int(self.sim["up"])
        num_syms = int(self.sim.get("num_syms_last", 1))

        settings = ChannelSettings(
            sample_rate=fs,
            number_of_taps=int(self.sim["num_taps"]),
            delay_spread=float(self.sim["delay_spread"]),
            snr_db=float(self.sim["snr_db"]),
        )
        mode = ChannelMode(multipath=1 if self.sim["multipath"] else 0, thermal_noise=1)

        try:
            channel = Channel_Model(settings, mode)
            rx_sig, _ = channel.apply(tx_sig)
            rx_sig = np.asarray(rx_sig).flatten()
            if np.mean(np.abs(rx_sig) ** 2) > 0:
                rx_sig *= np.sqrt(np.mean(np.abs(tx_sig) ** 2)) / np.sqrt(np.mean(np.abs(rx_sig) ** 2))
        except Exception as e:
            QMessageBox.critical(self, "Channel error", str(e))
            return

        try:
            rx = Receiver80211a(fs=fs, num_symbols=num_syms, nfft=64, ncp=16)
            corrected_symbols = rx.process_signal(rx_sig, tx_sig)
            rx_const = np.asarray(corrected_symbols).flatten() if corrected_symbols is not None else None
        except Exception as e:
            QMessageBox.critical(self, "RX error", str(e))
            return

        ax = self.plot_td.ax()
        _plot_td(ax, rx_sig, fs, "RX signal u vremenu (poslije kanala)")
        self.plot_td.draw()

        ax2 = self.plot_spec.ax()
        _plot_spec(ax2, rx_sig, fs, "RX spektar (normirano)")
        self.plot_spec.draw()

        ax3 = self.plot_const.ax()
        _plot_const(ax3, rx_const, "RX konstelacija (poslije prijemnika)")
        self.plot_const.draw()

        self.sim["rx_sig"] = rx_sig
        if "ch_sig" not in self.sim or self.sim["ch_sig"] is None:
            self.sim["ch_sig"] = rx_sig
        if "ch_const" not in self.sim or self.sim["ch_const"] is None:
            self.sim["ch_const"] = _extract_constellation_from_td(rx_sig, up, num_syms)
        self.sim["rx_const"] = rx_const

        self.info.setText("RX završeno.")

    def compare(self):
        if "tx_sig" not in self.sim or self.sim["tx_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo generiši TX.")
            return
        if "ch_sig" not in self.sim or self.sim["ch_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo primijeni Kanal ili pokreni RX.")
            return
        if "rx_sig" not in self.sim or self.sim["rx_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo pokreni RX.")
            return

        dlg = CompareDialog(self, "Uporedni prikaz (TX / Kanal / RX)")
        dlg.plot_3x3(
            self.sim["tx_sig"],
            self.sim["ch_sig"],
            self.sim["rx_sig"],
            float(self.sim["fs_last"]),
            tx_const=self.sim.get("tx_const", None),
            ch_const=self.sim.get("ch_const", None),
            rx_const=self.sim.get("rx_const", None),
        )
        dlg.exec_()


class SystemCanvas(QGraphicsView):
    def __init__(self, on_open, parent=None):
        super().__init__(parent)
        self.on_open = on_open
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setBackgroundBrush(QBrush(QColor("#ffffff")))
        self._build()

    def _add_block(self, title, key, x, y, w, h, color_hex):
        item = RoundedBlockItem(title, key, QRectF(x, y, w, h), QColor(color_hex), self.on_open)
        self.scene.addItem(item)
        return item

    def _wire(self, a, b):
        ra = a.path().boundingRect().translated(a.pos())
        rb = b.path().boundingRect().translated(b.pos())
        p1 = QPointF(ra.right(), ra.center().y())
        p2 = QPointF(rb.left(), rb.center().y())
        self.scene.addItem(WireItem(p1, p2))

    def _build(self):
        self.scene.clear()
        self.setSceneRect(0, 0, 1400, 720)

        title = self.scene.addText("802.11a Primopredajnik")
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        title.setFont(f)
        title.setDefaultTextColor(QColor("#111827"))
        title.setPos(240, 70)

        tx = self._add_block("TX", "tx", 170, 280, 300, 160, "#dbeafe")
        ch = self._add_block("Kanal", "channel", 550, 280, 300, 160, "#ffedd5")
        rx = self._add_block("RX", "rx", 930, 280, 300, 160, "#dcfce7")

        self._wire(tx, ch)
        self._wire(ch, rx)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.modules = _safe_import()
        self.setWindowTitle("802.11a Primopredajnik")

        self.sim = {
            "mod": "QPSK",
            "bps": 2,
            "up": 2,
            "payload_samples": 200_000,
            "snr_db": 20.0,
            "multipath": True,
            "num_taps": 2,
            "delay_spread": 10e-9,
            "tx_sig": None,
            "ch_sig": None,
            "rx_sig": None,
            "tx_const": None,
            "ch_const": None,
            "rx_const": None,
            "fs_last": 40e6,
            "num_syms_last": 1,
            "seed_last": None,
        }

        self.root = QWidget()
        root_lay = QVBoxLayout()
        root_lay.setContentsMargins(18, 18, 18, 18)
        root_lay.setSpacing(12)

        top = QWidget()
        top_lay = QHBoxLayout()
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(10)

        self.btn_setup = QPushButton("Podešavanja")
        self.btn_setup.clicked.connect(self.open_setup)

        self.btn_compare = QPushButton("Uporedi (3×3)")
        self.btn_compare.clicked.connect(self.open_compare)

        self.status = QLabel("Spremno.")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        top_lay.addWidget(self.btn_setup)
        top_lay.addWidget(self.btn_compare)
        top_lay.addStretch(1)
        top_lay.addWidget(self.status)
        top.setLayout(top_lay)

        self.canvas = SystemCanvas(self.open_block)

        root_lay.addWidget(top)
        root_lay.addWidget(self.canvas)
        self.root.setLayout(root_lay)
        self.setCentralWidget(self.root)
        self.resize(1500, 900)

        self._apply_style()

        if "error" in self.modules:
            QMessageBox.critical(self, "Import error", self.modules["error"])

        self._refresh_status()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #ffffff; }
            QLabel { color: #111827; font-size: 12px; }
            QPushButton {
                background: #111827;
                color: #ffffff;
                border: none;
                padding: 10px 14px;
                border-radius: 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #0b1220; }
            QPushButton:pressed { background: #060a12; }
            QTabWidget::pane { border: 1px solid #e5e7eb; border-radius: 12px; }
            QTabBar::tab {
                background: #f3f4f6;
                padding: 10px 14px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 6px;
                font-weight: 600;
            }
            QTabBar::tab:selected { background: #ffffff; }
            QSplitter::handle { background: #e5e7eb; }
            QDialog { background: #ffffff; }
            QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 7px 10px;
                border-radius: 10px;
                border: 1px solid #e5e7eb;
                background: #ffffff;
            }
            QCheckBox { padding: 6px 2px; }
            """
        )

    def _refresh_status(self):
        self.status.setText(
            f"Mod: {self.sim['mod']} | up: {self.sim['up']} | payload: {self.sim['payload_samples']} uzoraka | SNR: {self.sim['snr_db']:.1f} dB"
        )

    def open_setup(self):
        dlg = SetupDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            cfg = dlg.get_config()
            self.sim.update(cfg)
            self._refresh_status()

    def open_block(self, key):
        if "error" in self.modules:
            QMessageBox.critical(self, "Import error", self.modules["error"])
            return

        if key == "tx":
            dlg = TxDialog(self, self.modules, self.sim)
            dlg.exec_()
        elif key == "channel":
            dlg = ChannelDialog(self, self.modules, self.sim)
            dlg.exec_()
        elif key == "rx":
            dlg = RxDialog(self, self.modules, self.sim)
            dlg.exec_()

    def open_compare(self):
        if "tx_sig" not in self.sim or self.sim["tx_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo generiši TX.")
            return
        if "ch_sig" not in self.sim or self.sim["ch_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo primijeni Kanal (ili pokreni RX).")
            return
        if "rx_sig" not in self.sim or self.sim["rx_sig"] is None:
            QMessageBox.information(self, "Info", "Prvo pokreni RX.")
            return

        dlg = CompareDialog(self, "Uporedni prikaz (TX / Kanal / RX)")
        dlg.plot_3x3(
            self.sim["tx_sig"],
            self.sim["ch_sig"],
            self.sim["rx_sig"],
            float(self.sim["fs_last"]),
            tx_const=self.sim.get("tx_const", None),
            ch_const=self.sim.get("ch_const", None),
            rx_const=self.sim.get("rx_const", None),
        )
        dlg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
