package linkedIn;

import java.util.ArrayList;
import java.util.List;

public class Node {
	List<Node> children;
	char val;
	
	public Node(char val){
		this.val=val;
		children=new ArrayList<Node>();
	}
}
