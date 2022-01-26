from pylab import rcParams,rc, cycler
from matplotlib.ticker import ScalarFormatter

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"

cm = 1/2.54
page_width=15.4*cm
plot_height=5*cm
rcParams['figure.figsize'] = page_width/2 + 0.5*cm, plot_height + 0.5*cm
colors = cycler(color=["#0c5388", "#ffb400", "#1e8700", "#6f0023", "k"])
rcParams['axes.prop_cycle'] = colors
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
rcParams['font.size'] = 11
rcParams['figure.figsize'] = 9*cm, 5*cm
rcParams["figure.autolayout"] = True
