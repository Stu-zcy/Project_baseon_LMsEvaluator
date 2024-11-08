import logging

from prettytable import PrettyTable


class MyPrettyTable:
    def __init__(self):
        self.table = PrettyTable()

    def add_field_names(self, field_names):
        self.table.field_names = field_names

    def add_row(self, row):
        self.table.add_row(row)

    def print_table(self):
        print(self.table)

    def logging_table(self):
        logging.info("\n" + str(self.table))

    def set_align(self, col_name, align_type):
        self.table.align[col_name] = align_type


if __name__ == "__main__":
    table = PrettyTable()
    table.field_names = ['Name', 'Age', 'Country']
    table.add_row(['Alice', 26, 'USA'])
    table.add_row(['John', 30, 'UK'])
    table.align['Name'] = "l"
    print(table)
    # logging.error("\n" + str(table))
