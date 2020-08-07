package threadpermsg;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.Socket;

public class ClientHandler implements Runnable{
    private final Socket socket;
    private final String clientIdentify;
    public ClientHandler(Socket socket){
        this.socket=socket;
        this.clientIdentify=socket.getInetAddress().getHostAddress()+":"+socket.getPort();
    }

    @Override
    public void run() {
        // TODO Auto-generated method stub
        try {
            this.chat();
        } catch (IOException e) {
            //TODO: handle exception
            e.printStackTrace();
        }
    }
    private void chat() throws IOException{
        BufferedReader bufferedReader=wrap2Reader(this.socket.getInputStream());
        PrintStream printStream=warp2Print(this.socket.getOutputStream());
        String received;
        while((received=bufferedReader.readLine())!=null){
            System.out.println("cli"+clientIdentify+"msg:"+received);
            if(received.equals("quit")){
                write2Cli(printStream,"cli will close");
                socket.close();
                break;
            }
            write2Cli(printStream,"server"+received);
        }

    }

    private BufferedReader wrap2Reader(InputStream inputStream){
        return new BufferedReader(new InputStreamReader(inputStream));
    }
    private PrintStream warp2Print(OutputStream outputStream){
        return new PrintStream(outputStream);
    }
    private void write2Cli(PrintStream printStream,String msg) throws IOException{
        printStream.println(msg);
        printStream.flush();
    }
    

}
