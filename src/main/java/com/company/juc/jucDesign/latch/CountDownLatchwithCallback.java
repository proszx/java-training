package latch;

import java.util.concurrent.TimeUnit;

public class CountDownLatchwithCallback extends Latch{

    private Runnable runnable;

	public CountDownLatchwithCallback(int limit,Runnable runnable){
        super(limit);
        this.runnable=runnable;
    }


	@Override
	public void await() throws InterruptedException {
        // TODO Auto-generated method stub
        synchronized(this){
            if(limit>0){
                this.wait();
            }
        }
        if(null!=this.runnable){
            runnable.run();
        }
	}

	@Override
	public void await(TimeUnit unit, long time) throws InterruptedException, WaitTimeException {
		// TODO Auto-generated method stub
		if(time<=0){
            throw new IllegalArgumentException("time is invalid");
        }
        long remainNanos=unit.toNanos(time);
        final long endNanos=System.nanoTime()+remainNanos;
        synchronized(this){
            while(limit>0){
                if(TimeUnit.NANOSECONDS.toMillis(remainNanos)<=0){
                    throw new WaitTimeException("wait time over specify time.");
                }
                this.wait(TimeUnit.NANOSECONDS.toMillis(remainNanos));
                remainNanos=endNanos-System.nanoTime();
            }
        }
        if(null!=runnable){
            runnable.run();
        }
	}

	@Override
	public void countDown() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int getUnarrived() {
		// TODO Auto-generated method stub
		return 0;
	}


}