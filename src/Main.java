import java.util.*;

public class Main {
  public static void main(String[] args) {
    String a = "姓名fullName不能为空.证件类型不能为空电话号码不能为空";
    System.out.println(replaceAll(a));

  }

  public ArrayList<Integer> printListFromTailToHead(ListNode head) {
    ArrayList<Integer> list = new ArrayList<>();
    if (head == null) {
      return null;
    }
    ListNode reverseListNode = null;
    ListNode tempListNode = null;//store prior
    ListNode cuurListNode = head;//store current

    while (cuurListNode != null) {
      ListNode next = cuurListNode.next;//store the next

      //如果长度为1
      if (next == null) {
        reverseListNode = cuurListNode;
      }

      cuurListNode.next = tempListNode;
      tempListNode = cuurListNode;
      cuurListNode = next;
    }
    while (reverseListNode.next != null) {
      list.add(reverseListNode.next.val);
    }
    return list;
  }

  //二维数组的查找
  public static boolean Find(int target, int[][] array) {
    if (array.length > 0) {
      int rowCount = array.length;
      int columnCount = array[0].length;
      for (int i = rowCount - 1, j = 0; i >= 0 && j < columnCount; ) {
        if (target == array[i][j]) {
          return true;
        }
        if (target < array[i][j]) {
          i--;
          continue;
        }
        if (target > array[i][j]) {
          j--;
          continue;
        }
      }
    } else {
      return false;
    }
    return false;
  }

  //替换字符串
  public static String replaceSpace(StringBuffer str) {
    int originLength = str.length();
    int count = 0;
    for (int i = 0; i < str.length(); i++) {
      if (str.charAt(i) == ' ') {
        count++;
      }
    }
    return str.toString();
  }

  public static boolean Find1(int target, int[][] array) {
    int row = array.length - 1;
    //默认设置在最后一行
    //这样查找可以减少时间复杂度
    if (array.length > 0) {
      for (int i = array.length - 1; i >= 0; i--) {
        if (array[i].length > 0) {
          if (array[i][0] <= target && array[i][array[i].length - 1] >= target) {
            row--;
          }
        }
      }
    }
    for (int j = 0; j < array[row].length; j++) {
      if (array[row][j] == target) {
        return true;
      }
    }
    return false;
  }

  public static boolean FindFinal(int target, int[][] array) {
    //如果是++  从每一行最后一个算起
    //如果是--  从每一行第一个算起
    //移动的是整行整列   所以
    int row = array.length;
    int column = array[0].length;
    //默认设置在最后一行
    //这样查找可以减少时间复杂度
    int x = 0;
    int y = column - 1;
    while (x < row && y > 0) {
      if (array[x][y] > target) {
        y--;
      } else if (array[x][y] == target) {
        return true;
      } else {
        x--;
      }
    }
    return false;
  }

  //替换字符串
  public static String replaceSpace1(StringBuffer str) {
    StringBuffer stringBuffer = new StringBuffer();
    for (int i = 0; i < str.length(); i++) {
      if (str.charAt(i) == ' ') {
        stringBuffer.append("%20");
      } else {
        stringBuffer.append(str.charAt(i));
      }
    }
    return stringBuffer.toString();
  }

  Stack<Integer> stack1 = new Stack<Integer>();
  Stack<Integer> stack2 = new Stack<Integer>();

  public void push(int node) {
    stack2.push(node);
  }

  public int pop() {
    if (!stack2.empty()) {
      stack1.push(stack2.pop());
    }
    return stack1.pop();
  }

  public int minNumberInRotateArray(int[] array) {
    if (array == null || array.length == 0) {
      return 0;
    }
    int p1 = 0;//从前往后走
    int p2 = array.length - 1;//从后往前走
    int min = array[p1];//如果没发生旋转，直接将array[0]的值返回，
    int mid = 0;
    //当数组发生旋转了，
    while (array[p1] >= array[p2]) {
      //当两个指针走到挨着的位置时，p2就是最小数了
      if (p2 - p1 == 1) {
        min = array[p2];
        break;
      }
      mid = (p1 + p2) / 2;
      //如果中间位置的数既等于p1位置的数又等于P2位置的数
      if (array[p1] == array[mid] && array[p2] == array[mid]) {
        min = minInorder(array, p1, p2);
      }
      if (array[p1] <= array[mid])//若中间位置的数位于数组1，让p1走到mid的位置
      {
        p1 = mid;
      } else if (array[p2] >= array[mid])//若中间位置的数位于数组2，让p2走到mid的位置
      {
        p2 = mid;
      }
    }
    return min;
  }

  private int minInorder(int[] array, int p1, int p2) {
    int min = array[p1];
    for (int i = p1 + 1; i <= p2; i++) {
      if (min > array[i]) {
        min = array[i];
      }
    }
    return min;
  }

  public int Fibonacci(int n) {
    if (n < 0) {
      return -1;
    } else if (n == 0) {
      return 0;
    } else if (n == 1 || n == 2) {
      return 1;
    } else {
      return Fibonacci(n - 1) + Fibonacci(n - 2);
    }
  }

  public int JumpFloor(int target) {
    if (target <= 0) {
      return 0;
    } else if (target == 0) {
      return 0;
    } else if (target == 1) {
      return 1;
    } else if (target == 2) {
      return 2;
    } else {
      return JumpFloor(target - 1) + JumpFloor(target - 2);
    }
  }

  public int JumpFloorII(int target) {
    if (target <= 0) {
      return 0;
    } else if (target == 1) {
      return 1;
    } else {
      return JumpFloorII(target - 1) * 2;
    }
  }

  public static int RectOver(int target) {
    if (target <= 1) {
      return 1;
    } else if (target == 1) {
      return 1;
    } else if (target == 2) {
      return 2;
    } else {
      return RectOver(target - 1) + RectOver(target - 2);
    }
  }

  public static int NumberOf(int target) {
    int count = 0;
    while (target != 0) {
      ++count;
      target = (target - 1) & target;
    }
    return count;
  }

  public double Power(double base, int exponent) throws Exception {
    double result = 0.0;
    if ((0.0 == base) && exponent < 0) {
      throw new Exception("0的负次幂没有意义！");
    } else if (exponent < 0) {
      return 1 / powerWithExponent(base, -exponent);
    } else {
      return powerWithExponent(base, exponent);
    }
  }

  public double powerWithExponent(double base, int exponent) {
    if (exponent == 0) {
      return 1;
    }
    if (exponent == 1) {
      return base;
    }
    double result = 1.0;
    for (int i = 0; i < exponent; i++) {
      result = result * base;
    }
    return result;
  }

  public static ListNode FindKthToTail(ListNode head, int k) {
    ListNode listNode = null;
    if (head == null) {
      return null;
    } else {
      if (k <= 0) {
        return null;
      }
      if (k == 1) {
        return head;
      }
      head = head.next;
      FindKthToTail(head, k - 1);
    }
    return null;
  }


  public static TreeNode reConstructBinaryTree(int[] pre, int[] in) {
    TreeNode treeNode = null;
    if (pre == null || pre.length == 0 || in == null || in.length == 0) {
      return null;
    }

    if (pre.length == 1 && in.length == 1) {
      return new TreeNode(pre[0]);
    }
    for (int i = 0; i < in.length; i++) {
      if (in[i] == pre[0]) {
        treeNode = new TreeNode(in[i]);
        treeNode.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, i + 1), Arrays.copyOfRange(in, 0, i));
        treeNode.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i + 1, pre.length), Arrays.copyOfRange(in, i + 1, in.length));
      }
    }
    return treeNode;
  }

  public ListNode ReverseList(ListNode head) {
    ListNode cuur = head;
    ListNode next = null;
    ListNode pre = null;
    if (head == null || head.next == null) {
      return head;
    } else {
      while (cuur != null) {
        next = cuur.next;
        cuur.next = pre;
        pre = cuur;
        cuur = next;
      }
    }
    return pre;
  }

  public ListNode Merge(ListNode list1, ListNode list2) {
    ListNode listNode = null;
    if (list1 == null) {
      return list1;
    }
    if (list2 == null) {
      return list2;
    }
    if (list1 == null && list2 == null) {
      return null;
    }
    if (list1.val < list2.val) {
      listNode = list1;
      listNode.next = Merge(list1.next, list2);
    } else {
      listNode = list2;
      listNode.next = Merge(list1, list2.next);
    }
    return listNode;
  }


  public boolean isSubtree(TreeNode root1, TreeNode root2) {
    if (root2 == null) {
      return true;
    }
    if (root1 == null) {
      return false;
    }
    if (root1.val == root2.val) {
      return isSubtree(root1.left, root2.left) && isSubtree(root1.right, root2.right);
    } else {
      return false;
    }
  }

  public void Mirror(TreeNode root) {
    TreeNode treeNode = null;
    if (root == null) {
      return;
    }
    treeNode = root.left;
    root.left = root.right;
    root.right = treeNode;
    Mirror(root.left);
    Mirror(root.right);
  }


  public static ArrayList<Integer> printMatrix(int[][] matrix) {
    ArrayList<Integer> arrayList = new ArrayList<>();
    int beginRow = 0;
    int beginColumn = 0;
    int row = matrix.length;
    int column = matrix[0].length;
    while (true) {
      for (int i = beginColumn; i < column; i++) {
        arrayList.add(matrix[beginRow][i]);
      }
      beginRow++;
      if (beginRow >= row) {
        break;
      }

      for (int i = beginRow; i < row; i++) {
        arrayList.add(matrix[i][column - 1]);
      }
      column--;
      if (beginColumn >= column) {
        break;
      }

      for (int i = column - 1; i >= beginColumn; i--) {
        arrayList.add(matrix[row - 1][i]);
      }
      row--;
      if (beginRow >= row) {
        break;
      }

      for (int i = row - 1; i >= beginRow; i--) {
        arrayList.add(matrix[i][beginColumn]);
      }
      beginColumn++;
      if (beginColumn >= column) {
        break;
      }
    }
    return arrayList;
  }

  public boolean HasSubtree(TreeNode root1, TreeNode root2) {
    boolean result = false;
    if (root1 != null && root2 != null) {
      if (root1.val == root2.val) {
        result = AhaseB(root1, root2);
      }
      if (result == false) {
        result = HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
      }
    }
    return result;
  }

  public boolean AhaseB(TreeNode root1, TreeNode root2) {
    if (root2 == null)
      return true;
    if (root1 == null)
      return false;
    if (root1.val == root2.val) {
      return AhaseB(root1.left, root2.left) && AhaseB(root1.right, root2.right);
    }
    return false;
  }

  public static boolean IsPopOrder(int[] pushA, int[] popA) {
    if (pushA.length == 0 || popA.length == 0) {
      return false;
    }
    int popIndex = 0;
    Stack<Integer> auxiliaryStack = new Stack<>();
    for (int i = 0; i < pushA.length; i++) {
      auxiliaryStack.push(pushA[i]);
      while (!auxiliaryStack.empty() && auxiliaryStack.peek() == popA[popIndex]) {
        auxiliaryStack.pop();
        popIndex++;
      }
    }
    return auxiliaryStack.empty();
  }

  public static ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
    ArrayList<Integer> arrayList = new ArrayList<>();
    LinkedList<TreeNode> linkedList = new LinkedList<>();
    if (root == null) {
      return arrayList;
    }
    linkedList.add(root);
    while (linkedList.size() != 0) {
      TreeNode tmp = linkedList.remove(0);
      if (tmp.left != null) {
        linkedList.add(tmp.left);
      }
      if (tmp.right != null) {
        linkedList.add(tmp.right);
      }
      arrayList.add(tmp.val);
    }
    return arrayList;
  }

  public boolean VerifySquenceOfBST(int[] sequence) {
    if (sequence == null || sequence.length < 0) {
      return false;
    }
    int root = sequence[sequence.length - 1];
    int i;
    for (i = 0; i < sequence.length; i++) {
      if (sequence[i] > root) {
        break;
      }
    }
    int j;
    for (j = i; j < sequence.length - 1; j++) {
      if (sequence[j] < root) {
        return false;
      }
    }
    boolean left = true;
    if (i > 0) {
      left = VerifySquenceOfBST(Arrays.copyOfRange(sequence, 0, i));
    }
    boolean right = true;
    if (i < sequence.length - 1) {
      right = VerifySquenceOfBST(Arrays.copyOfRange(sequence, i, sequence.length - 1));
    }
    return left && right;
  }

  public boolean verifySequenceOfBST(int[] array) {
    if (array == null || array.length <= 0)
      return false;
    int root = array[array.length - 1];
    int i = 0;
    for (; i < array.length - 1; ++i) {
      if (array[i] > root) {
        break;
      }
    }
    int j = i;
    for (; j < array.length - 1; ++j) {
      if (array[j] < root) {
        return false;
      }
    }

    boolean leftFlag = true;

    if (i > 0) {

      leftFlag = verifySequenceOfBST(Arrays.copyOfRange(array, 0, i));

    }

    boolean rightFlag = true;

    if (i < array.length - 1) {

      rightFlag = verifySequenceOfBST(Arrays.copyOfRange(array, i, array.length - 1));

    }

    return leftFlag && rightFlag;
  }

  static ArrayList<ArrayList<Integer>> paths = new ArrayList();
  static ArrayList<Integer> list = new ArrayList<>();
  static int sum = 0;

  public static ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
    if (root == null) {
      return paths;
    }
    sum += root.val;
    list.add(root.val);
    if (sum == target && root.left == null && root.right == null) {
      // 存放路径结点值
      ArrayList<Integer> path = new ArrayList<Integer>();
      for (int i = 0; i < list.size(); i++) {
        path.add(list.get(i));
      }
      paths.add(path);
    }
    //先左后右   因为同一级左边总是小于右边
    if (sum < target && root.left != null) {
      FindPath(root.left, target);
    }
    if (sum < target && root.right != null) {
      FindPath(root.right, target);
    }
    sum -= root.val;
    list.remove(list.size() - 1);
    return paths;
  }

  public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
    ArrayList<ArrayList<Integer>> combinitions = new ArrayList<>();
    if (sum < 3) {
      return combinitions;
    }
    int small = 1;
    int big = 2;
    int curSum = small + big;
    int middle = (1 + sum) / 2;
    while (small < middle) {
      ArrayList<Integer> combinition = new ArrayList<Integer>();
      if (curSum == sum) {
        for (int i = small; i <= big; i++) {
          combinition.add(i);
        }
      }
      while (curSum > sum && small < middle) {
        curSum -= small;
        small++;
        if (curSum == sum) {
          for (int j = small; j <= big; j++) {
            combinition.add(j);
          }
        }
      }
      if (combinition.size() > 0) combinitions.add(combinition);
      big++;
      curSum += big;
    }
    return combinitions;
  }

  public static ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
    ArrayList<Integer> result = new ArrayList<>();
    if (array.length < 2) {
      return result;
    }
    int begin = 0;
    int end = array.length - 1;
    while (begin < end) {
      if (array[begin] + array[end] == sum) {
        result.add(array[begin]);
        result.add(array[end]);
        break;
      } else if (array[begin] + array[end] > sum) {
        end--;
      } else {
        begin++;
      }
    }
    return result;
  }

  public static String LeftRotateString(String str, int n) {
    if (str == null || str.length() < 2 || str.length() < n) {
      return str;
    }
    char[] array = str.toCharArray();
    reverse(array, 0, n - 1);
    reverse(array, n, array.length - 1);
    reverse(array, 0, array.length - 1);
    return String.valueOf(array);
  }

  //反转数组
  public static char[] reverse(char[] array, int start, int end) {
    char temp = ' ';
    while (start < end) {
      temp = array[start];
      array[start++] = array[end];
      array[end--] = temp;
    }
    return array;
  }

  public String ReverseSentence(String str) {
//    if (str == null || str.length() < 2) {
//      return str;
//    }
//    char[] array = reverse(str.toCharArray(), 0, str.length() - 1);
//    int begin = 0;
//    int end = 0;
//    while (begin < end && end <= array.length) {
//      if ("" == array[end]) {
//
//      }
//    }
//  }
    return null;
  }


  public static class RandomListNode {
    String label;
    RandomListNode next = null;
    RandomListNode random = null;

    RandomListNode(String label) {
      this.label = label;
    }
  }

  public RandomListNode Clone(RandomListNode pHead) {
    return null;

  }


  public static void copyList(RandomListNode head) {

    //不断地循环插入
    while (head != null) {
      RandomListNode copyNode = new RandomListNode(head.label);
      //将新创建的节点放到两个节点之间
      copyNode.next = head.next;
      copyNode.random = null;
      head.next = copyNode;

      //接着复制下一个节点
      head = copyNode.next;
    }
  }

  public static void setSbiling(RandomListNode head) {
    //遍历给其赋值random
    while (head != null) {
      RandomListNode copyNode = head.next;
      if (head.random != null) {
        copyNode.random = head.random.next;
      }

      //接着复制下一个节点
      head = copyNode.next;
    }
  }

  private static RandomListNode ReconnectNodes(RandomListNode pHead) {
    if (pHead == null)
      return null;
    RandomListNode tmpNode = pHead;
    RandomListNode newHead = pHead.next;

    while (tmpNode != null) {
      RandomListNode node = tmpNode.next;
      tmpNode.next = node.next;
      if (node.next != null)
        node.next = node.next.next;
      else
        node.next = null;
      tmpNode = tmpNode.next;
    }

    return newHead;
  }


//
//  public static TreeNode baseconvert(TreeNode root, TreeNode lastNode) {
//    if (root == null)
//      return lastNode;
//    TreeNode current = root;
//    if (current.left != null)
//      lastNode=baseconvert(current.left, lastNode);
//    current.left = lastNode;
//    if (lastNode != null)
//      lastNode.right = current;
//    lastNode = current;
//    if (current.right != null)
//      lastNode=baseconvert(current.right, lastNode);
//    return lastNode;
//  }
//
//  public static TreeNode convert(TreeNode root) {
//    TreeNode lastNode = null;
//    lastNode=baseconvert(root, lastNode);
//    TreeNode headNode = lastNode;
//    while (headNode.left != null)
//      headNode = headNode.left;
//    return headNode;
//
//  }

  //二叉树转换为双向链表
  public TreeNode baseconvert(TreeNode root, TreeNode lastNode) {
    if (root == null)
      return lastNode;
    TreeNode current = root;
    if (current.left != null)
      lastNode = baseconvert(current.left, lastNode);
    current.left = lastNode;
    if (lastNode != null)
      lastNode.right = current;
    lastNode = current;
    if (current.right != null)
      lastNode = baseconvert(current.right, lastNode);
    return lastNode;
  }

  public TreeNode convert(TreeNode pRootOfTree) {
    TreeNode lastNode = null;
    lastNode = baseconvert(pRootOfTree, lastNode);
    TreeNode headNode = lastNode;
    while (headNode != null && headNode.left != null)
      headNode = headNode.left;
    return headNode;

  }


  public static int solve(int idx, int[] nums) {
    if (idx < 0) {
      return 0;
    }
    return Math.max(nums[idx] + solve(idx - 2, nums), solve(idx - 1, nums));
  }

  public int rob(int[] nums) {
    return solve(nums.length - 1, nums);
  }

  public static ArrayList<String> Permutation(String str) {
    ArrayList<String> result = new ArrayList<String>();//根据返回类型需要
    if (str == null || str.length() == 0) {
      return result;
    }
    char[] chars = str.toCharArray();
    TreeSet<String> res = new TreeSet<String>(); //用于排序输出
    getResult(chars, 0, str.length() - 1, res);
    result.addAll(res);//添加到ArrayList
    return result;
  }

  //从最外层起  一直进行递归  外层循环保证    每个数字都被交换
  public static void getResult(char[] chars, int start, int end, TreeSet<String> res) {

    if (start == end) {
      res.add(String.valueOf(chars));
    } else {
      for (int i = start; i <= end; i++) {
        swap(chars, start, i);//换一位
        getResult(chars, start + 1, end, res);//递归
        swap(chars, start, i);//换回来，保证下次换位是正确的
      }
    }
  }


  //交换两个字符串的位置
  public static void swap(char[] chars, int a, int b) {
    if (a == b) {//因为会出现原位置与原位置交换，直接空即可

    } else {
      char temp = chars[a];
      chars[a] = chars[b];
      chars[b] = temp;
    }
  }

  public int getMiddle(Integer[] list, int low, int high) {
    int tmp = list[low];    //数组的第一个作为中轴
    while (low < high) {
      while (low < high && list[high] > tmp) {
        high--;
      }
      list[low] = list[high];   //比中轴小的记录移到低端
      while (low < high && list[low] < tmp) {
        low++;
      }
      list[high] = list[low];   //比中轴大的记录移到高端
    }
    list[low] = tmp;              //中轴记录到尾
    return low;                   //返回中轴的位置
  }

  public void quickSort(Integer[] list, int low, int high) {
    if (low < high) {
      int middle = getMiddle(list, low, high);  //将list数组进行一分为二
      quickSort(list, low, middle - 1);        //对低字表进行递归排序
      quickSort(list, middle + 1, high);       //对高字表进行递归排序
    }
  }

  static int morethanhalf2(int[] array) {
    int re = array[0];
    int num = 1;
    int count = 0;
    for (int i = 1; i < array.length; i++) {
      if (num == 0) {
        re = array[i];
        num++;
      } else {
        if (re == array[i])
          num++;
        else
          num--;
      }
    }
    for (int i = 0; i < array.length; i++) {
      if (re == array[i]) {
        count++;
      }
    }
    if (count > array.length / 2) {
      return re;
    } else {
      return 0;
    }
  }

  public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
    if (input == null) {
      return null;
    }
    ArrayList<Integer> list = new ArrayList<>(k);
    if (k > input.length)
      return list;
    TreeSet<Integer> tree = new TreeSet<Integer>();
    for (int i = 0; i < input.length; i++) {
      tree.add(input[i]);
    }
    int i = 0;
    for (Integer elem : tree) {
      if (i >= k)
        break;
      list.add(elem);
      i++;
    }
    return list;
  }

  //动态规划思想初步
  public int FindGreatestSumOfSubArray(int[] array) {
    int preSum = 0;
    int max = Integer.MIN_VALUE;
    for (int i = 0; i < array.length; i++) {
      //前部分和为负数，便不要
      if (preSum < 0) {
        preSum = array[i];
      } else {
        //前部分和为正数，便要
        preSum = preSum + array[i];
      }
      max = Math.max(preSum, max);
    }
    return max;
  }

  //求出某个整数区间中1的个数
  public static int NumberOf1Between1AndN_Solution(int n) {
    if (n <= 0) {
      return 0;
    }
    int count = 0;
    int factor = 1;
    while (n / factor != 0) {
      int lowerNum = n - n / factor * factor;
      int currentNum = (n / factor) % 10;
      int highNum = n / (factor * 10);
      if (currentNum == 0) {
        // 如果为0,出现1的次数由高位决定
        count += highNum * factor;
      } else if (currentNum == 1) {
        // 如果为1,出现1的次数由高位和低位决定
        count += highNum * factor + lowerNum + 1;
      } else {
        // 如果大于1,出现1的次数由高位决定
        count += (highNum + 1) * factor;
      }
      factor *= 10;
    }
    return count;
  }

  public static String PrintMinNumber(int[] numbers) {
    ArrayList<Integer> list = new ArrayList<>();
    for (int val : numbers) {
      list.add(val);
    }
    Collections.sort(list, new Comparator<Integer>() {
      //根据Comparable指定顺序对list集合排序
      public int compare(Integer str1, Integer str2) {
        String s1 = str1 + "" + str2;
        //str1和str2都是整数，用 str1 +"" + str2即将整数转化为了字符串
        String s2 = str2 + "" + str1;
        return s1.compareTo(s2);
      }
    });
    StringBuilder sBuilder = new StringBuilder();
    for (int value : list) {
      sBuilder.append(String.valueOf(value));
    }
    return sBuilder.toString();
  }

  public static Boolean isUglyNum(Integer value) {
    Boolean isUglyNum = false;
    if (value < 0) {
      return false;
    }
    if (value == 1 || value == 2 || value == 3) {
      return true;
    }
    for (int i = 2; i < value; i++) {

    }
    return isUglyNum;
  }

  public static int GetUglyNumber_Solution(int index) {
    if (index <= 0)
      return 0;
    if (index < 7)
      return index;
    int[] res = new int[index];
    res[0] = 1;
    int t2 = 0,//记录乘以2的个数
        t3 = 0,//记录乘以3的个数
        t5 = 0,//记录乘以5的个数
        i = 1;
    for (; i < index; i++) {
      res[i] = Math.min(res[t2] * 2, Math.min(res[t3] * 3, res[t5] * 5));
      if (res[i] == res[t2] * 2) t2++;
      //如果最小的等于res[t2]*2，那么t2加1
      if (res[i] == res[t3] * 3) t3++;
      if (res[i] == res[t5] * 5) t5++;
    }
    return res[index - 1];
  }

  //找出第一个只出现一次的字符
  public static int FirstNotRepeatingChar(String str) {
    if (str == null || str.length() < 1) {
      return -1;
    }
    char[] chars = str.toCharArray();
    Map<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < str.length(); i++) {
      Character val = str.charAt(i);
      if (!map.containsKey(str.charAt(i))) {
        map.put(val, 1);
      } else {
        map.put(val, map.get(val) + 1);
      }
    }
    for (int i = 0; i < str.length(); i++) {
      if (map.get(chars[i]) == 1) {
        return i;
      }
    }
    return -1;
  }

  /**
   *
   * @param array
   * @return
   */
  public static int iPairs(int[] array) {
    if (array == null)
      throw new IllegalArgumentException();
    // 创建辅助数组
    int length = array.length;
    int[] copy = new int[length];
    System.arraycopy(array, 0, copy, 0, length);
    int numberOfInversePairs = iPairs(array, copy, 0, length - 1);
    return numberOfInversePairs;
  }



  /**
   *
   * 功能描述:
   *
   * @param:
   * @return:
   * @auther: delicate
   * @date: 2018/7/20 11:20
   */
  public static int iPairs(int[] array, int[] copy, int begin, int end) {
    if (begin == end)
      return 0;
    int mid = (begin + end) / 2;
    // 递归调用
    int left = iPairs(copy, array, begin, mid);
    int right = iPairs(copy, array, mid + 1, end);

    // 归并
    int i = mid, j = end, pos = end;
    int count = 0; // 记录相邻子数组间逆序数

    while (i >= begin && j >= mid + 1) {
      if (array[i] > array[j]) {
        copy[pos--] = array[i--];
        count += j - mid;
      } else
        copy[pos--] = array[j--];
    }

    while (i >= begin)
      copy[pos--] = array[i--];
    while (j >= mid + 1)
      copy[pos--] = array[j--];

    return left + right + count;
  }

  /**
   * 求两个链表的第一个公共节点
   * @param pHead1
   * @param pHead2
   * @return
   */
  public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
    if(pHead1 == null || pHead2 == null) {
      return null;
    }
    Map<Integer,ListNode> map = new HashMap<>();

    while (pHead1 != null) {
      if (!map.containsKey(pHead1.val)) {
        map.put(pHead1.val,pHead1);
      }
      pHead1 = pHead1.next;
    }

    while (pHead2 != null) {
      if (map.containsKey(pHead2.val)) {
        return pHead2;
      }
      pHead2 = pHead2.next;
    }
    return null;
  }

  /**
   * 统计一个数组在排序数组中出现的次数
   * @param array
   * @param k
   * @return
   */
  public static int GetNumberOfK(int [] array , int k) {
    {
      int number = 0;
      if (array != null && array.length > 0)
      {
        int first = GetFirstK(array, k, 0, array.length - 1);
        int last = GetLastK(array, k, 0, array.length - 1);

        if (first > -1 && last > -1)
        {
          number = last - first + 1;
        }
      }
      return number;
    }
  }
  public static int GetFirstK(int[] data, int k, int start, int end)
  {
    if (start > end)
    {
      return -1;
    }

    int middIndex = (start + end) / 2;
    int middData = data[middIndex];

    if (middData == k)
    {
      if ((middIndex > 0 && data[middIndex - 1] != k) || middIndex == 0)
      {
        return middIndex;
      }
      else
      {
        end = middIndex - 1;
      }
    }
    else if (middData > k)
    {
      end = middIndex - 1;
    }
    else
    {
      start = middIndex + 1;
    }

    return GetFirstK(data, k, start, end);
  }

  public static int  GetLastK(int[] data, int k, int start, int end)
  {
    if (start > end)
    {
      return -1;
    }

    int middIndex = (start + end) / 2;
    int middData = data[middIndex];

    if (middData == k)
    {
      if ((middIndex < data.length - 1 && data[middIndex + 1] != k) || middIndex == end)
      {
        return middIndex;
      }
      else
      {
        start = middIndex + 1;
      }
    }
    else if (middData > k)
    {
      end = middIndex - 1;
    }
    else
    {
      start = middIndex + 1;
    }

    return GetLastK(data, k, start, end);
  }

  /**
   * 求二叉树的深度
   * @param root
   * @return
   */
  public static int TreeDepth(TreeNode root) {
    if (root == null)
      return 0;
    else {
      int left = TreeDepth(root.left);
      int right = TreeDepth(root.right);
      return 1 + Math.max(left, right);
    }
  }

  /**
   * 找出数组中两个只出现一次数字
   * @param array
   * @param num1
   * @param num2
   */
  public static void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {

    if (array == null || array.length == 0) {
      num1 = null;
      num2 = null;
    }

    Map<Integer,Integer> map = new HashMap<>();
    for (Integer val : array) {
      if (!map.containsKey(val)) {
        map.put(val,null);
      } else {
        map.remove(val);
      }
    }

    List<Integer> list = new ArrayList<Integer>();
    list.addAll(map.keySet());

    for (int i=0;i<list.size();i++) {
      if (i==0) {
        num1[0] = list.get(i);
      }
      if (i==1) {
        num2[0] = list.get(i);
      }
    }


  }

//  public String ReverseSentence(String str) {
//    if (str.trim().equals("") || str == null) {
//      return str;
//    }
//    // 第一步：翻转整个句子
//    String sentenceReverse = reverse(str);
//    String[] spilt = sentenceReverse.split(" ");
//    String result = "";
//    // 第二步：翻转每个单词
//    for (String i : spilt) {
//      result = result + reverse(i) + " ";
//    }
//    // 删除最后一个空格
//    result = result.substring(0, result.length() - 1);
//    return String.valueOf(result);
//  }


  //将整个字符串进行翻转
//  private String reverse(String str) {
//
//    char[] arr = str.toCharArray();
//    char temp;
//    for (int i = 0; i < arr.length / 2; i++) {
//      temp = arr[i];
//      arr[i] = arr[arr.length - i - 1];
//      arr[arr.length - i - 1] = temp;
//    }
//    return String.valueOf(arr);
//  }

private static String replaceAll(String errDesc){
  if (errDesc.indexOf(".") != -1) {
   return errDesc.replaceAll("."," ");
  } else {
    return errDesc;
  }
}




}


