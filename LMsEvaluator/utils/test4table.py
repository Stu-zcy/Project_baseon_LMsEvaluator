import logging
import time
from tqdm import tqdm
from prettytable import PrettyTable

# table = PrettyTable()
#
# table.field_names = ['Name', 'Age', 'Country']
#
# table.add_row(['Alice', 26, 'USA'])
# table.add_row(['John', 30, 'UK'])
#
# print(table)
# logging.error("\n" + str(table))

if __name__ == '__main__':
    # acc = 0.12345
    # acc_format = "{:.3%}".format(acc)
    # print(acc_format)
    # temp = tqdm(range(100))
    # for i in temp:
    #     temp.set_description(f"Processing {i}")
    #     time.sleep(.01)

    table = PrettyTable()

    # table.field_names = ['\033[1mSummary\033[0m', '']
    #
    # table.add_row(['\033[1mpoison_dataset:\033[0m', '\033[1msst-2\033[0m'])
    # table.add_row(['\033[1mpoisoner:\033[0m', '\033[1mbadnets\033[0m'])
    # table.add_row(['\033[1mpoison_rate:\033[0m', '\033[1m0.1\033[0m'])
    # table.add_row(['\033[1mlabel_consistency:\033[0m', '\033[1mno\033[0m'])
    # table.add_row(['\033[1mlabel_dirty:\033[0m', '\033[1mno\033[0m'])
    # table.add_row(['\033[1mtarget_label\033[0m', '\033[1m1\033[0m'])
    # table.add_row(['\033[1mCACC:\033[0m', '\033[1m0.91543\033[0m'])
    # table.add_row(['\033[1mASR:\033[0m', '\033[1m1\033[0m'])
    # table.add_row(['\033[1m∆PPL\033[0m', '\033[1m351.36\033[0m'])
    # table.add_row(['\033[1m∆GE:\033[0m', '\033[1m0.70833\033[0m'])
    # table.add_row(['\033[1mUSE:\033[0m', '\033[1m0.93209\033[0m'])
    # table.align['Summary'] = "l"
    # table.align[''] = "l"
    # print(table)

    table.field_names = ['\033[1mResults\033[0m', '']
    table.add_row(['\033[1mOriginal accuracy\033[0m', f"\033[1m{(0.8 * 100):.2f}%\033[0m"])
    table.add_row(['\033[1mPoisoning accuracy\033[0m', f"\033[1m{(0 * 100):.2f}%\033[0m"])
    table.align['Results'] = "l"
    print(table)
