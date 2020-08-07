package single;
public class FlightSecurity{
    private int cnt=0;
    private String boarding="null";
    private String idCard="null";

    public synchronized void pass(String boarString,String idCaString){
        this.boarding=boarString;
        this.idCard=idCaString;
        this.cnt++;
        check();

    }
    private void check(){
        if(boarding.charAt(0)!=idCard.charAt(0)){
            throw new RuntimeException("message"+toString());
        }
    }
    public String toString(){
        StringBuffer sb=new StringBuffer();
        sb.append("The ")
        .append(cnt)
        .append("passenger,boardings")
        .append(boarding)
        .append("idcard is")
        .append(idCard);
        return sb.toString();
    }
}