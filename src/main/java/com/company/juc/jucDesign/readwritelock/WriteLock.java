package readwritelock;

public class WriteLock implements Lock{
    private ReadWriteLockImpl readWriteLock;
    WriteLock(ReadWriteLockImpl readWriteLock){
        this.readWriteLock=readWriteLock;
    }
    @Override
    public void lock() throws InterruptedException {
        // TODO Auto-generated method stub
        synchronized(readWriteLock.getMutex()){
           try {
               //写锁 需要先放入等待区进行的古代
               readWriteLock.increWaitingWriters();
               while(readWriteLock.getReadingWriters()>0||readWriteLock.getWritingWriters()>0){
                   readWriteLock.getMutex().wait();
               } 
           } finally {
               this.readWriteLock.decWaitingWriter();
           }
           readWriteLock.increWritingWriters();
        }
    }

    @Override
    public void unlock() {
        // TODO Auto-generated method stub
        synchronized(readWriteLock.getMutex()){
            readWriteLock.decWritingWriter();
            readWriteLock.changePrefer(false);
            readWriteLock.getMutex().notifyAll();
        }
    }

}
