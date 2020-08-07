package twophase;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.*;
public class ChatServer {
    private final int port;
    private ThreadPoolExecutor threadPoolExecutor;
    private ServerSocket serverSocket;
    public ChatServer(int port){
        this.port = port;
    }
    public ChatServer(){
        this(13322);
    }
    public void startServer()throws IOException{
        this.threadPoolExecutor=new ThreadPoolExecutor(1,4,0, null, null);
        this.serverSocket=new ServerSocket(port);
        this.serverSocket.setReuseAddress(true);
        System.out.println("Chat server is online in"+port);
        this.listen();
    }
    private void listen() throws IOException{
        while(true){
            Socket socket=serverSocket.accept();
            this.threadPoolExecutor.execute(new ClientHandler(socket));
        }
    }
    public static void main(String[] args) throws IOException {
        new ChatServer(8079).startServer();
    }
    
}