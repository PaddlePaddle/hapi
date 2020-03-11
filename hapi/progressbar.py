import sys
import time
import numpy as np


class ProgressBar(object):
    """progress bar """

    def __init__(self,
                 num=None,
                 width=30,
                 verbose=1,
                 start=True,
                 file=sys.stdout):
        self._num = num
        if isinstance(num, int) and num <= 0:
            raise TypeError('num should be None or integer (> 0)')
        max_width = self._get_max_width()
        self._width = width if width <= max_width else max_width
        self._total_width = 0
        self._verbose = verbose
        self.file = file
        self._completed = 0
        self._values = {}
        self._values_order = []
        if start:
            self._start = time.time()
        self._last_update = 0

        self._dynamic_display = (
            (hasattr(self.file, 'isatty') and
             self.file.isatty()) or 'ipykernel' in sys.modules or
            'posix' in sys.modules or 'PYCHARM_HOSTED' in os.environ)

    def _get_max_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_width = min(int(terminal_width * 0.6), terminal_width - 50)
        return max_width

    def start(self):
        self.file.flush()
        self._start = time.time()

    def update(self, current_num, values=None):
        self._completed = current_num

        now = time.time()
        info = ' - elapsed %.0fs' % (now - self._start)
        if self._verbose == 1:
            prev_total_width = self._total_width

            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self._num is not None:
                numdigits = int(np.log10(self._num)) + 1

                bar_chars = ('step %' + str(numdigits) + 'd/%d [') % (
                    current_num, self._num)
                prog = float(current_num) / self._num
                prog_width = int(self._width * prog)

                if prog_width > 0:
                    bar_chars += ('=' * (prog_width - 1))
                    if current_num < self._num:
                        bar_chars += '>'
                    else:
                        bar_chars += '='
                bar_chars += ('.' * (self._width - prog_width))
                bar_chars += ']'
            else:
                bar_chars = 'step %5d' % current_num

            self._total_width = len(bar_chars)
            sys.stdout.write(bar_chars)

            if current_num:
                time_per_unit = (now - self._start) / current_num
            else:
                time_per_unit = 0

            if self._num is not None and current_num < self._num:
                eta = time_per_unit * (self._num - current_num)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) //
                                                   60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, 'step')
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, 'step')
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, 'step')

            for k, v in values:
                info += ' - %s:' % k
                if isinstance(v, (float, np.float32, np.float64)):
                    if abs(v) > 1e-3:
                        info += ' %.4f' % v
                    else:
                        info += ' %.4e' % v
                else:
                    info += ' %s' % v
            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            # newline for another epoch
            if self._num is not None and current_num >= self._num:
                info += '\n'
            if self._num is None:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()
            self._last_update = now
        elif self._verbose == 2:
            #if self._num is not None and current_num >= self._num:
            if self._num:
                numdigits = int(np.log10(self._num)) + 1
                count = ('step %' + str(numdigits) + 'd/%d') % (current_num,
                                                                self._num)
            else:
                count = 'step %5d' % current_num
            info = count + info
            for k, v in values:
                info += ' - %s:' % k
                if isinstance(v, (float, np.float32, np.float64)):
                    if abs(v) > 1e-3:
                        info += ' %.4f' % v
                    else:
                        info += ' %.4e' % v
                elif isinstance(v, np.ndarray) and \
                     isinstance(v.size, 1) and \
                     isinstance(v.dtype, (np.float32, np.float64)):
                    if abs(v[0]) > 1e-3:
                        info += ' %.4f' % v[0]
                    else:
                        info += ' %.4e' % v[0]
                else:
                    info += ' %s' % v
            info += '\n'
            sys.stdout.write(info)
            sys.stdout.flush()
