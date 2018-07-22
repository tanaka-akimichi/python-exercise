from stop_watch import stop_watch

@stop_watch
def func() :
    j=0
    for i in range(99999999) :
        j+=i
    print(j)

func()

