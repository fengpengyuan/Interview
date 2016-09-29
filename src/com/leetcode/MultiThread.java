package com.leetcode;

public class MultiThread implements Runnable{
	private int ticketNum=0;
	private boolean flag=true;
	private synchronized void sell(){
		if(ticketNum<100){
			System.out.println(Thread.currentThread().getName()+" is selling "+ticketNum);
			ticketNum++;
		}
		else
			flag=false;
	}
	@Override
	public void run(){
		while(flag){
			sell();
//			synchronized(this){
//				if(ticketNum<50){
//					System.out.println(Thread.currentThread().getName()+" is selling "+ticketNum);
//					ticketNum++;
//				}
//				else
//					flag=false;
//			}
			
			
			try{
				Thread.sleep(2000);
			}catch(InterruptedException e){
				e.printStackTrace();
			}
		}
	}
	
	
	public static void main(String[] args){
		MultiThread ticket=new MultiThread();
		Thread th1 = new Thread(ticket,"window 1");
		Thread th2 = new Thread(ticket,"window 2");
		Thread th3 = new Thread(ticket,"window 3");
		Thread th4 = new Thread(ticket,"window 4");
		
		th1.start();
		th2.start();
		th3.start();
		th4.start();
	}

}
