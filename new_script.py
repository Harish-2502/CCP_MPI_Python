import time
import json
from datetime import datetime
from datetime import timezone
from zoneinfo import ZoneInfo
from mpi4py import MPI
start_time = time.time()

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

def time_interval(time):
    time = int(time)
    if time==0:
        return f'{12-time}am - {time+1}am'
    elif time==11:
        return f'{time}am - {time+1}pm'
    elif time==12:
        return f'{time}pm - {time-12+1}pm'
    elif time>12 and time<23:
        return f'{time-12}pm - {time-12+1}pm'
    elif time==23:
        return f'{time-12}pm - {time-12+1}am'
    else:
        return f'{time}am - {time+1}am'

def date_time(date_time):
    Melbourne_tzone = ZoneInfo("Australia/Melbourne")
    d = datetime.fromisoformat(
    date_time[:-1]
    ).astimezone(Melbourne_tzone)

    return d.strftime('%Y-%m-%d %H:%M:%S')

def date(date_time):
    Melbourne_tzone = ZoneInfo("Australia/Melbourne")
    d = datetime.fromisoformat(
    date_time[:-1]
    ).astimezone(Melbourne_tzone)

    return d.strftime('%Y-%m-%d')

def hour(date_time):
    Melbourne_tzone = ZoneInfo("Australia/Melbourne")
    d = datetime.fromisoformat(
    date_time[:-1]
    ).astimezone(Melbourne_tzone)

    return d.strftime('%Y-%m-%d %H')

def cvt_data(a):
    try:
        a = json.loads(str(a[:-2]))
    except:
        a = {}
    return a

def happiest_hour(final_data):
    for data in final_data:
        if data != {} and 'sentiment' in data['doc']['data'] and type(data['doc']['data']['sentiment'])!=dict and type(float(data['doc']['data']['sentiment']))==float:
            if hour(data['doc']['data']['created_at']) in happy_hour_dict:
                happy_hour_dict[hour(data['doc']['data']['created_at'])] += float(data['doc']['data']['sentiment'])
            else:
                happy_hour_dict[hour(data['doc']['data']['created_at'])] = float(data['doc']['data']['sentiment'])

def happiest_day(final_data):
    for data in final_data:
        if data != {} and 'sentiment' in data['doc']['data'] and type(data['doc']['data']['sentiment'])!=dict and type(float(data['doc']['data']['sentiment']))==float:
            if date(data['doc']['data']['created_at']) in happy_date_dict:
                happy_date_dict[date(data['doc']['data']['created_at'])] += float(data['doc']['data']['sentiment'])
            else:
                happy_date_dict[date(data['doc']['data']['created_at'])] = float(data['doc']['data']['sentiment'])

def most_active_hour(final_data):
    for data in final_data:
        if data != {}:
            if hour(data['doc']['data']['created_at']) in mtweets_hour_dict:
                mtweets_hour_dict[hour(data['doc']['data']['created_at'])] += 1
            else:
                mtweets_hour_dict[hour(data['doc']['data']['created_at'])] = 1

def most_active_day(final_data):
    for data in final_data:
        if data != {}:
            if date(data['doc']['data']['created_at']) in mtweets_date_dict:
                mtweets_date_dict[date(data['doc']['data']['created_at'])] += 1
            else:
                mtweets_date_dict[date(data['doc']['data']['created_at'])] = 1

happy_hour_dict = {}
happy_date_dict = {}
mtweets_hour_dict = {}
mtweets_date_dict = {}

chunk_size = 50000
with open("/home/harishk/twitter-100gb.json") as f:
    count = 0
    while True:
        #print(count)
        try:
            data = [next(f) for _ in range(chunk_size)]
            data = [cvt_data(i) for i in data]
            if count%size == rank:
                happiest_hour(data)
                happiest_day(data)
                most_active_hour(data)
                most_active_day(data)
            count += 1
            if not data:
                break
        except StopIteration:
            break

happy_hour_dict = comm.gather(happy_hour_dict, root=0)
happy_date_dict = comm.gather(happy_date_dict, root=0)
mtweets_hour_dict = comm.gather(mtweets_hour_dict, root=0)
mtweets_date_dict = comm.gather(mtweets_date_dict, root=0)

print(happy_hour_dict)
if rank == 0:

    hhour_dict = {}
    for j in happy_hour_dict:
        for key,value in j.items():
            if key in hhour_dict:
                hhour_dict[key] += value
            else:
                hhour_dict[key] = value

    hdate_dict = {}
    for j in happy_date_dict:
        for key,value in j.items():
            if key in hdate_dict:
                hdate_dict[key] += value
            else:
                hdate_dict[key] = value

    mhour_dict = {}
    for j in mtweets_hour_dict:
        for key,value in j.items():
            if key in mhour_dict:
                mhour_dict[key] += value
            else:
                mhour_dict[key] = value

    mdate_dict = {}
    for j in mtweets_date_dict:
        for key,value in j.items():
            if key in mdate_dict:
                mdate_dict[key] += value
            else:
                mdate_dict[key] = value

    # happiest hour ever in the data
    happy_hour_dict = dict(sorted(hhour_dict.items()))
    happy_hour_dict = max(happy_hour_dict.items(), key=lambda k: k[1])
    print(f'The happiest hour ever is between {time_interval(happy_hour_dict[0].split()[1])} on {datetime.fromisoformat(happy_hour_dict[0].split()[0]).strftime("%d %B, %Y")} with an overall sentiment score of {happy_hour_dict[1]}')

    # happiest day ever in the data
    happy_date_dict = dict(sorted(hdate_dict.items()))
    happy_date_dict = max(happy_date_dict.items(), key=lambda k: k[1])
    print(f'{datetime.fromisoformat(happy_date_dict[0]).strftime("%d %B, %Y")} was the happiest day with an overall sentiment score of {happy_date_dict[1]}')

    # most active hour ever
    mtweets_hour_dict = dict(sorted(mhour_dict.items()))
    mtweets_hour_dict = max(mtweets_hour_dict.items(), key=lambda k: k[1])
    print(f'The most active hour ever is between {time_interval(mtweets_hour_dict[0].split()[1])} on {datetime.fromisoformat(mtweets_hour_dict[0].split()[0]).strftime("%d %B, %Y")} with {mtweets_hour_dict[1]} tweets')

    # most active day ever
    mtweets_date_dict = dict(sorted(mdate_dict.items()))
    mtweets_date_dict = max(mtweets_date_dict.items(), key=lambda k: k[1])
    print(f'{datetime.fromisoformat(mtweets_date_dict[0]).strftime("%d %B, %Y")} was the most active day with with {mtweets_date_dict[1]} tweets')
    print("--- %s seconds ---" % (time.time() - start_time))
