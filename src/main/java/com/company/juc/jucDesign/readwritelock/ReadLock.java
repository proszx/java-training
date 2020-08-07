package readwritelock;

public class ReadLock implements Lock{
    private final ReadWriteLockImpl readWriteLockImpl;
    ReadLock(ReadWriteLockImpl readWriteLock){
        this.readWriteLockImpl=readWriteLock;
    }
    @Override
    public void lock() throws InterruptedException {
        // TODO Auto-generated method stub
        synchronized(readWriteLockImpl.getMutex()){
            while(readWriteLockImpl.getWritingWriters()>0||(readWriteLockImpl.getPreferWriter()&&readWriteLockImpl.getWaitingWriters()>0)){
                readWriteLockImpl.getMutex().wait();
            }
            readWriteLockImpl.increWritingWriters();
        }
    }

    @Override
    public void unlock() {
        // TODO Auto-generated method stub
        synchronized(readWriteLockImpl.getMutex()){
            readWriteLockImpl.decReadingWriter();
            readWriteLockImpl.changePrefer(true);
            readWriteLockImpl.getMutex().notifyAll();
        }
    }

}
