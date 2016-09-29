package linkedIn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class ConstantRandomGet<E> {
	HashMap<E, Integer> map;
	List<E> list;
	
	public ConstantRandomGet(){
		map=new HashMap<E, Integer>();
		list=new ArrayList<E>();
	}
	
	public void insert(E e){
		map.put(e, list.size());
		list.add(e);
	}
	
	public boolean remove(E e){
		if(!map.containsKey(e))
			return false;
		E lastE=list.get(list.size()-1);
		int index=map.get(e);
		list.remove(list.size()-1);
		list.set(index, lastE);
		map.put(lastE, index);
		return true;
	}
	
	public boolean contains(E e){
		return map.containsKey(e);
	}
	
	public E getRandom(){
		int size=list.size();
		if(size==0)
			return null;
		Random r=new Random();
		int index=r.nextInt(size);
		return list.get(index);
	}
}
