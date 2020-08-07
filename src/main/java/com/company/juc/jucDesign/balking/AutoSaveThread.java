package com.company.juc.jucdesign.balking;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class AutoSaveThread  extends Thread{
	private final Document document;
	
	public AutoSaveThread(final Document document) {
		super("Document AutoSaveThread");
		this.document=document;
	}
	@Override
	public void run(){
		while(true){
			try {
				document.save();
				TimeUnit.SECONDS.sleep(1);
			} catch (IOException|InterruptedException e) {
				//TODO: handle exception
				e.printStackTrace();
			}
		}
	}


}
