package readwritelock;

public class ReadWriteLockImpl  implements ReadWriteLock{
    private final Object Mutex=new Object();
    private int writingWriters=0;
    private int waitingWriters=0;
    private int readingWriters=0;
    private boolean perferWriter;
    public ReadWriteLockImpl(){
        this(true);
    }
    
    public ReadWriteLockImpl(boolean perferWriter) {
        this.perferWriter=perferWriter;
    }
    
	@Override
    public Lock readLock() {
        // TODO Auto-generated method stub
        return new ReadLock(this);
    }

    @Override
    public Lock writeLock() {
        // TODO Auto-generated method stub
        return new WriteLock(this);
    }
    void increWritingWriters(){
        this.writingWriters++;
    }
    void increWaitingWriters(){
        this.waitingWriters++;
    }
    void increReadingWriter(){
        this.readingWriters++;
    }
    void decWritingWriter(){
        this.writingWriters--;
    }
    void decWaitingWriter(){
        this.waitingWriters--;
    }
    void decReadingWriter(){
        this.readingWriters--;
    }
    @Override
    public int getWritingWriters() {
        // TODO Auto-generated method stub
        return this.writingWriters;
    }

    @Override
    public int getWaitingWriters() {
        // TODO Auto-generated method stub
        return this.waitingWriters;
    }

    @Override
    public int getReadingWriters() {
        // TODO Auto-generated method stub
        return this.readingWriters;
    }
    Object getMutex(){
        return this.Mutex;
    }
    boolean getPreferWriter(){
        return this.perferWriter;
    }
    void changePrefer(boolean perferWriter){
        this.perferWriter=perferWriter;
    }

}
