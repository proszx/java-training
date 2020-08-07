package com.company.juc.hook;

import java.io.IOException;
import java.nio.file.*;
import java.util.concurrent.TimeUnit;


//Hook的具体应用
public class PreventDup {
    final  static  String Path="~/.lock";
    final static  String file=".lock";



    public static void main(String[] args) throws IOException {
        Runtime.getRuntime().addShutdownHook(new Thread(){
            @Override
            public void run() {
                System.out.println("get the shutdown signal");
                getLockFile().toFile().delete();
            }
        });
        checkRunning();
        while(true){
            try{
                TimeUnit.MILLISECONDS.sleep(100);
                System.out.println("running");
            }catch (InterruptedException e){
                e.printStackTrace();
            }
        }
    }
    private static Path getLockFile(){
        return Paths.get(Path,file);
    }
    private static  void checkRunning() throws IOException {
        Path path=getLockFile();
        if(path.toFile().exists()){
            throw  new RuntimeException("already running");
        }
        Files.createFile(path);
    }
}
