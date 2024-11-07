from datetime import datetime


def convert_to_fp():
    # value = 738818
    date = '2023-10-25'
    value_ord = datetime.strptime(date, '%Y-%m-%d').toordinal()
    fp = to_fp(value_ord)
    print(f"date: {date} --> ord: {value_ord} --> fp: {fp}")


def convert_from_fp(mag):
    # value = 738818
    # mag = 48419176448
    value_ord = int(from_fp(mag))
    date = datetime.fromordinal(value_ord).strftime('%Y-%m-%d')
    print(f"fp: {mag} --> ord: {value_ord} --> date: {date}")

def date_to_ordinal(date):
    return datetime.strptime(date, '%Y-%m-%d').toordinal()

def to_fp(value, fp_impl='FP16x16'):
    sign = 0 if value >= 0 else 1

    match fp_impl:
        case 'FP16x16':
            return (abs(int(value * 2**16)), sign)
        case 'FP8x23':
            return (abs(int(value * 2**23)), sign)
        case 'FP32x32':
            return (abs(int(value * 2**32)), sign)
        case 'FP64x64':
            return (abs(int(value * 2**64)), sign)


def from_fp(value, fp_impl='FP16x16'):
    match fp_impl:
        case 'FP16x16':
            return value / 2**16
        case 'FP8x23':
            return value / 2**23
        case 'FP32x32':
            return value / 2**32
        case 'FP64x64':
            return value / 2**64


if __name__ == "__main__":
    # ord = date_to_ordinal('2023-10-25')
    # convert_to_fp()
    convert_from_fp(429495729)
    convert_from_fp(431461809)