package com.company.juc.jucdesign.balking;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * balking模式 多个线程监控某个共享变量，A即将触发动作，B已经行动，A放弃 自旋锁
 */
public class Document {
    private boolean isChange=false;
    private List<String> content=new ArrayList<>();
    private final FileWriter writer;
    private static AutoSaveThread autoSaveThread;
    private Document(String path,String name)throws IOException{
        this.writer=new FileWriter(new File(path,name), true);
    }
    public static Document create(String path,String name) throws IOException{
        Document document=new Document(path,name);
        autoSaveThread=new AutoSaveThread(document);
        autoSaveThread.start();
        return document;
    }
    public void edit(String content){
        synchronized(this){
            this.content.add(content);
            this.isChange=true;
        }
    }
    public void close() throws IOException{
        autoSaveThread.interrupt();
        writer.close();
    }
    public void save() throws IOException{
        synchronized(this){
            if(!isChange){
                return;
            }
            System.out.println(Thread.currentThread()+"execute the save action");
            
            for(String cacheLine:content){
                this.writer.write(cacheLine);
                this.writer.write("\r\n");
            }
            this.writer.flush();
            this.isChange=false;
            this.content.clear();
        }
    }
    
}