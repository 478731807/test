#coding=utf8
import collections
import sys
import time

def marquee(length=50, speed=1, direction=1):
    """
    生成一个简单的跑马灯
    length:总长
    speed：每0.1秒的移动速度
    direction:0为向左，1为向右
    """
    if direction == 1:
        array = '>'
    else:
        array = '<'
    que = collections.deque([array])
    que.extend(['-'] * (length - 1)) # 形如'>------'
    while True:
        print('%s' % ''.join(que))
        if direction == 1:
            que.rotate(1 * speed)
        else:
            que.rotate(-1 * speed)
        sys.stdout.flush()
        time.sleep(0.1)

if '__main__' == __name__:
    marquee()
