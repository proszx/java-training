package workerthread;

public abstract class Introduction {
    public final void create(){
        this.firstProcess();
        this.secondProcess();
    }
    protected abstract void firstProcess();
    protected  abstract void secondProcess();
}