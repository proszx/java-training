package single;
public class FlightSecurityTest{
    static class Passengers extends Thread{
        private final FlightSecurity flightSecurity;
        private final String idCard;
        private final String boarding;
        public Passengers(FlightSecurity flightSecurity,String idCard, String boarding){
            this.flightSecurity=flightSecurity;
            this.idCard=idCard;
            this.boarding=boarding;
        }
        @Override
        public void run(){
            while(true){
                flightSecurity.pass(boarding, idCard);
            }
        }
    }
    public static void main(String[] args) {
        final FlightSecurity flightSecurity=new FlightSecurity();
        new Passengers(flightSecurity, "A123456", "AF123456").start();
        new Passengers(flightSecurity, "B123456", "BF123456").start();
        new Passengers(flightSecurity, "C123456", "CF123456").start();

    }
}