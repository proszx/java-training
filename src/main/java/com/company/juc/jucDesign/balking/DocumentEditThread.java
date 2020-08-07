package com.company.juc.jucdesign.balking;
import java.io.IOException;
import java.util.Scanner;

public class DocumentEditThread extends Thread{
    private final String path;
    private final String name;
    private final Scanner sc=new Scanner(System.in);
    public DocumentEditThread(String path,String name){
        super("DocumentEditThread");
        this.path=path;
        this.name=name;
    }
    @Override
    public void run(){
        int times=0;
        try {
            Document document=Document.create(path, name);
            while(true){
                String text=sc.next();
                /**
                 * balking
                 * if(flag){
                 *  exit
                 * }else{
                 *  do sth
                 * this.flag=true;
                 * return ;
                 * }
                */
                if("quit".equals(text)){
                    document.close();
                    break;
                }
                document.edit(text);
                if(times==6){
                    document.save();
                    times=0;
                }
                times++;
            }
        } catch (IOException e) {
            //TODO: handle exception
            throw new RuntimeException(e);
        }
    }
}