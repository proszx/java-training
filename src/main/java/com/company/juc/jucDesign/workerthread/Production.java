package workerthread;

public class Production extends Introduction{
    private final int prodId;
    public Production(int prodId){
        this.prodId = prodId;
    }

	@Override
	protected void firstProcess() {
		// TODO Auto-generated method stub
		System.out.println("execute"+prodId+"first time");
	}

	@Override
	protected void secondProcess() {
        // TODO Auto-generated method stub
        
		System.out.println("execute"+prodId+"second time");
		
	}

}