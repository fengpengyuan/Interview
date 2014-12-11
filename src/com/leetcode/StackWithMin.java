package com.leetcode;

import java.util.Stack;

public class StackWithMin {
	private Stack<Integer> stack;
    private Stack<Integer> minStack;
     
    public StackWithMin(){
        stack = new Stack<Integer>();
        minStack = new Stack<Integer>();
    }
     
    /**
     * Add an element to the top of a stack
     * @param data
     */
    public void push(int data){
        //case 1. stack is empty
        stack.push(data);
        if(minStack.isEmpty())
        	minStack.push(data);
        else if(data<=minStack.peek())
        	minStack.push(data);
        
    }
    /**
     * remove the top element of a stack
     * @return
     * @throws Exception
     */
    public int pop() throws Exception{
        //edge case: stack is empty, throw an exception
        if(stack.isEmpty())
            throw new Exception("Error: stack is empty.");
        int data = stack.pop();
        //case 1: minStack is not empty
//        System.out.println("data is "+data);
        if( data==minStack.peek())
        	minStack.pop();
//            System.out.println(minStack.pop()+" pop out");
        return data;
    }
    /**
     * Return the current min value in a stack
     * @return
     * @throws Exception 
     */
    public int min() throws Exception{
    	if(minStack.isEmpty())
    		throw new Exception("stk is empty!");
    	else
    		return minStack.peek();
//        return minStack.isEmpty()?stack.peek():minStack.peek();
    }
     
    public static void main(String[] args){
        try{
            StackWithMin stack = new StackWithMin();
            stack.push(7);
            stack.push(10);
            stack.push(9);
            stack.push(8);
            stack.push(7);
            stack.push(6);
            stack.push(8);
            stack.push(5);
            stack.push(4);
            stack.push(6);
            stack.push(1);
             
             
            System.out.println(stack.min());//1
            stack.pop();
            System.out.println(stack.min());//4
            stack.pop();
            System.out.println(stack.min());//5
            stack.pop();
            System.out.println(stack.min());//6
            stack.pop();
            System.out.println(stack.min());//6
            stack.pop();
            System.out.println(stack.min());//7
            stack.pop();
            System.out.println(stack.min());//8
            stack.pop();
            System.out.println(stack.min());//9
            stack.pop();
            System.out.println(stack.min());//10
 
        }catch(Exception e){
            System.out.println(e);
        }
    }
}