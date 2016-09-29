package com.leetcode2;

public class TurtleMovement {
	int curX;
	int curY;
	//0-N, 1-E, 2-S, 3-W
	int direction;
	public TurtleMovement(){
		curX=0;
		curY=0;
		direction=0;
	}
	
	public void forward(){
		if(direction==0)
			curY++;
		else if(direction==1){
			curX++;
		}else if(direction==2)
			curY--;
		else
			curX--;
	}
	
	public void turnRight(){
		if(direction==0){
			direction=1;
			curX++;
		}else if(direction==1){
			direction=2;
			curY--;
		}else if(direction==2){
			direction=3;
			curX--;
		}else if(direction==3){
			direction=0;
			curY++;
		}			
	}
	
	public void getCoordinates(){
		System.out.println("("+curX+","+curY+")");
	}
	
	public static void main(String[] args){
		TurtleMovement turtle=new TurtleMovement();
		String input="FFFFR";
		
		for(int i=0;i<input.length();i++){
			char c=input.charAt(i);
			if(c=='F'){
				turtle.forward();
			}else
				turtle.turnRight();
		}
		
		turtle.getCoordinates();
	}
}
