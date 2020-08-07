package com.company.juc.jucdesign.activeobject.general;

public class ActiveDaemonThread extends Thread {
    private ActiveMessageQueue activeQueue;
    public ActiveDaemonThread(ActiveMessageQueue activeQueue){
        super("ActiveDeamonThread");
        this.activeQueue=activeQueue;
        setDaemon(true);
    }

	@Override
    public void run(){
        while(true){
            ActiveMessage methodMessage=this.activeQueue.take();
            methodMessage.execute();
        }
    }
}
