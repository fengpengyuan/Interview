package com.leetcode;

public class Employee {
	public Employee subordinate;
	public Employee superior;
	public String name;
	
	public Employee(String name){
		this.name=name;
	}
	
	public static void printAllNames(Employee e){
		printAllNames(e, "");
	}
	
	public static void printAllNames(Employee e, String s){
		if(e==null)
			return;
		System.out.println(s+e.name);
		s+="\t";
		printAllNames(e.subordinate,s);
		
	}
	
	
	public static void main(String args[]){
		Employee boss=new Employee("boss");
		Employee e1=new Employee("employee1");
		Employee e2 =new  Employee("employee2");
		Employee e3 =new  Employee("employee3");
		Employee e4 =new  Employee("employee4");
		boss.subordinate=e1;
		e1.superior=boss;
		
		e1.subordinate=e2;
		e2.superior=e1;
		
		e2.subordinate=e3;
		e3.superior=e2;
		
		e3.subordinate=e4;
		e4.superior=e3;
		e4.subordinate=null;
		
		printAllNames(boss);
		
	}
	
}
