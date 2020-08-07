package com.company.juc.singleton;

import java.net.Socket;
import java.sql.Connection;

public class Singleton {
    //饿汉
    static  final class Single{
        private byte[] data=new byte[1024];
        private static  Single instance=new Single();
        private  Single(){

        }
        private Single getInstance(){
            return instance;
        }
    }

    static  final class Singlel{
        private byte[] data=new byte[1024];
        private static  Singlel instance;
        private  Singlel(){

        }
        private Singlel getInstance(){
            if(instance==null){
                instance=new Singlel();
            }
            return instance;
        }
    }
    static  final  class Singlell{
        private byte[] data=new byte[1024];
        private  static  Singlell instance;
        private Singlell(){

        }
        private synchronized   Singlell getInstance(){
            if(instance==null){
                instance=new Singlell();
            }
            return instance;
        }
    }

    static  final  class SingletonDoubleCheck{
        private byte[] data=new byte[1024];
        private static SingletonDoubleCheck instance;
        Connection conn;
        Socket socket;
        private SingletonDoubleCheck(Connection conn,Socket socket){
            this.conn=conn;
            this.socket=socket;
        }
        private SingletonDoubleCheck getInstance(){
            if(instance==null){
                synchronized (SingletonDoubleCheck.class){
                    if(instance==null){
                        instance=new SingletonDoubleCheck(conn,socket);
                    }
                }
            }
            return instance;
        }
    }
    static  final  class SingletonDoubleCheckVola{
        private byte[] data=new byte[1024];
        private static volatile SingletonDoubleCheckVola instance;
        Connection conn;
        Socket socket;
        private SingletonDoubleCheckVola(Connection conn,Socket socket){
            this.conn=conn;
            this.socket=socket;
        }
        private SingletonDoubleCheckVola getInstance(){
            if(instance==null){
                synchronized (SingletonDoubleCheckVola.class){
                    if(instance==null){
                        instance=new SingletonDoubleCheckVola(conn,socket);
                    }
                }
            }
            return instance;
        }
    }
    static  final  class SingletonHolder{
        private byte[] data=new byte[1024];

        private SingletonHolder(){

        }
        private static class Holders{
            //有点像饿汉式
            private static  SingletonHolder instance=new SingletonHolder();
        }
        private static  SingletonHolder getInstance(){
            return Holders.instance;
        }
    }
    //饿汉式
    //懒汉式
    //懒汉式 sync
    //懒汉double check
    //懒汉double check volatile
    //枚举
    //枚举懒加载

}
