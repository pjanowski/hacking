from datetime import datetime

def parse_text(filename):
    with open(filename) as f:
        lines = f.readlines()
        slots = []
        for line in lines:
            slot = line.strip().split(';')
            date = slot[0].split(':')[-1].strip()
            time = slot[1].split('ime:')[-1].strip()
            slot = [date, time]
            slots.append(slot)
    return slots

def is_Saturday_or_Sunday(date: str) -> bool:
    date = datetime.strptime(date, '%d/%m/%Y')
    return date.weekday() == 5 or date.weekday == 6

def main(filename: str) -> bool:
    slots = parse_text(filename)

    # are there 10 slots
    if len(slots) != 10:
        print(f"ERROR: {len(slots)} slots found, expected 10")
        # return False
    for slot in slots:

        hour = slot[1].split(':')[0]
        min = slot[1].split(':')[1]
        month = slot[0].split('/')[1]   

        # is each slot on a Sat /Sun
        if month not in ['05', '06']:
            print(f"ERROR: {slot} is not in May or June")
            # return False

        # is each slot at top of hour or half
        if int(hour) < 8 or int(hour) > 16:
            print(f"ERROR: {slot} is not between 8 and 4")
            # return False

        # test if the date is a Saturday
        if slot[0].split('/')[0] == 'Sat':
            print(f"ERROR: {slot} is not on a Saturday")
            # return False

        # is each slot on a Sat /Sun
        date = slot[0]
        if not is_Saturday_or_Sunday(date):
            print(f"ERROR: {slot} is not on a Saturday or Sunday")
            # return False

    # is each slot at top of hour or half
    # at most one slot at 7:30

main('data.txt')
