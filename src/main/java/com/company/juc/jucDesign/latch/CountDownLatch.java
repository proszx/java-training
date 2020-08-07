package latch;

import java.util.concurrent.TimeUnit;

public class CountDownLatch extends Latch{

    public CountDownLatch(int limit){
        super(limit);
    }
	@Override
	public void await() throws InterruptedException {
        // TODO Auto-generated method stub
        synchronized(this){
            while(limit>0){
                this.wait();
            }
        }
		
	}
    @Override
    public void await(TimeUnit unit,long time) throws InterruptedException,WaitTimeException{
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
    }
	@Override
	public void countDown() {
		// TODO Auto-generated method stub
		synchronized(this){
            if(limit<=0){
                throw new IllegalStateException("all of task arrived");
            }
            limit--;
            this.notifyAll();
        }
	}

	@Override
	public int getUnarrived() {
		// TODO Auto-generated method stub
		return limit;
	}

}