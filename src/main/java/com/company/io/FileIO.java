package com.company.io;

import java.io.*;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

public class FileIO {
    static  void copyFile(String sou,String des) throws IOException {
         FileInputStream in=new FileInputStream(sou);
        FileOutputStream out=new FileOutputStream(des);
        byte[] buffer=new byte[1024*20];
        int cnt;
        while((cnt=in.read(buffer,0,1024*20))!=-1){
            out.write(buffer,0,buffer.length);
        }
        in.close();
        out.close();
    }
    //逐行读取
    static  void readLines(String sou)throws  IOException{
        FileReader fb=new FileReader(sou);
        BufferedReader bufferedInputStream=new BufferedReader(fb);
        String line;
        while((line=bufferedInputStream.readLine())!=null){
            System.out.println(line);
        }
    }
    static  void nioCopy(String sou,String des) throws IOException {
        FileInputStream in=new FileInputStream(sou);

        FileOutputStream out=new FileOutputStream(des);

        FileChannel inchannel=in.getChannel();
        FileChannel outchannel=out.getChannel();

        ByteBuffer buffer=ByteBuffer.allocateDirect(1024);

        while (true){
            int r=inchannel.read(buffer);
            if(r==-1){
                break;
            }
            buffer.flip();
            outchannel.write(buffer);
            buffer.clear();
        }
    }
    public static void main(String[] args) {

    }
}
