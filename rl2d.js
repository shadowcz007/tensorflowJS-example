Array.prototype.randomChooseOne = function () {
  var num = this.length;
  var index = Math.ceil(Math.random() * num) - 1;
  return this[index];

};

function Actions(_names, _actions, _actionsByState) {

  this.names = _names;

  this.action = 0;


  var actionsByState = _actionsByState;

  this.actions = tf.tensor1d(_actions);

  this.actionsByState = actionsByState;

};

Actions.prototype.max = function () {
  this.length = this.names.length;
  return this.length;
}

Actions.prototype.getName = function (_action) {
  var _a = _action || this.action;
  this.name = this.names[_a];
  return this.name;
};

Actions.prototype.randomChooseOne = function (_state) {

  var _actions = this.actionsByState[_state];

  var _action = _actions.randomChooseOne();

  return _action;

};


function States(_states, _rewards) {
  var states = _states;
  var rewards = _rewards;

  this.states = states;
  this.rewards = rewards;

  this.max = function () {
    return this.states.length;
  };
};



States.prototype.randomChooseOne = function () {
  this.state = this.states.randomChooseOne();
  return this.state;
};

States.prototype.nextState = function (_action, _state) {

  //U	D L	R	N

  var n_state;
  switch (_action) {
    case 0:
      n_state = _state - 2;
      break;
    case 1:
      n_state = _state + 2;
      break;
    case 2:
      n_state = _state - 1;
      break;

    case 3:
      n_state = _state + 1;
      break;

    default:
      n_state = _state;

  };

  return n_state;


};


States.prototype.getReward = function (_state_n) {

  var reward;
  for (var i = 0; i < this.max(); i++) {
    if (this.states[i] == _state_n) {
      reward = this.rewards[i];
      return reward;
    }
  }

};



function QTable(statesLn, actionsLn) {
  var q_table = tf.zeros([statesLn, actionsLn]).buffer();
  this.q_table = q_table;

  console.log('===========================init');
  q_table.toTensor().print();

};

QTable.prototype.getFutureValue = function (state_n, actions_n) {
  //future value

  var q_table = this.q_table;


  var qs = [];

  for (var i = 0; i < actions_n.length; i++) {
    qs.push(q_table.get(state_n, actions_n[i]));
  };

  var res = tf.tensor1d(qs).max().dataSync()[0];

  return res;

};

QTable.prototype.update = function (state, action, reward, futureValue) {

  //var value = reward +GAMMA * tf.tensor1d(qs).max().dataSync()[0];
  //console.log(value)
  /*
    简化版
    眼前利益reward，和记忆中的利益qs
    value=reward+GAMMA*max(qs)
  */


  var value = (1 - ALPHA) * this.q_table.get(state, action) + ALPHA * (reward + GAMMA * futureValue);
  /*

    学习速率ALPHA越大，保留之前训练的效果就越少

  */

  this.q_table.set(value, state, action);
};

QTable.prototype.print = function () {
  console.log('---q_table');
  this.q_table.toTensor().print();
};



//step 设定
const GAMMA = 0.7, //折扣因子
  EPISODES_MAX = 100, //最大回合数
  ALPHA = 0.1; //学习率


//step0 定义actions,states,rewards

const actions_array = [0, 1, 2, 3, 4],
  actions_labels = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE'],
  actions_byState = [
    [1, 3, 4],
    [1, 2, 4],
    [0, 3, 4],
    [0, 2, 4]
  ],
  states_array = [0, 1, 2, 3],
  rewards_array = [0, 0, -10, 20];

var actions = new Actions(actions_labels, actions_array, actions_byState);
var states = new States(states_array, rewards_array);


//step1 初始化Q-table矩阵

var Q = new QTable(states_array.length, actions_array.length);

//q_table = tf.zeros([states.max(), actions.max()]).buffer();
//game = tf.zeros([states.max(), actions.max()]).buffer();



//step2 选择起始state`

var state = states.randomChooseOne();


//step3 选择当前state(s)下的一个可能action(a)
var action = actions.randomChooseOne(state);

/*
选择action的策略是随机,Q-learning并非每次迭代都沿当前Q值最高的路径前进。
*/

var episode = 0;

while (state != 3 || episode < EPISODES_MAX) {
  console.log('===========================');
  console.log('episode:' + episode + ',state:' + state);

  //step4 换移到下一个state(s')
  var state_n = states.nextState(action, state);

  //game.set(episode + 1, state, action);
  //game.toTensor().print();


  //step5 使用Bellman Equation，更新Q-table
  var reward = states.getReward(state_n);

  var actions_n = actions.actionsByState[state_n];

  var futureValue = Q.getFutureValue(state_n, actions_n);

  Q.update(state, action, reward, futureValue);

  //step6 将下一个state作为当前state
  state = state_n;
  action = actions.randomChooseOne(state);

  episode++;

  Q.print();

};

/*
for(var i=0;i<EPISODES_MAX;i++){

	//step4 换移到下一个state(s')
    var state_n = nextState(action, state);
    game.set(i+1,state,action);
	game.toTensor().print();

	//step5 使用Bellman Equation，更新Q-table
    updateQ(state, action, state_n);

	console.log("当前状态:" + state + ",动作:" + actions_name[action] + ",下一状态:" + state_n)
	//step6 将下一个state作为当前state
	state=state_n;
	action = chooseAction(state);

	if(state==3){
		console.log('SUCCESS');
		break;
	};

};
*/
console.log('===========================done');
Q.print();



/*

ALPHA=1
GAMMA=0.7

[[0         , 27.8149796, 0         , 34.0214005, 14.6019993],
     [0         , 48.6020012, 23.8149796, 0         , 34.0214005],
     [23.8149796, 0         , 0         , 60.4704857, 27.8149796],
     [34.0214005, 0         , 27.8149796, 0         , 57.8149796]]


ALPHA=0.1
GAMMA=0.7

[[0        , -4.9318953, 0         , 2.7061532, 0.5565188],
     [0        , 12.264082 , 0.4945346 , 0        , 3.1886277],
     [0.5898073, 0         , 0         , 7.2662129, -3.364475],
     [2.5542755, 0         , -2.3881776, 0        , 4.0108719]]


ALPHA=0.1
GAMMA=0.2
     [[0        , -4.6246333, 0         , 0.7205356 , 0.0144107 ],
     [0        , 10.5041332, 0.024998  , 0         , 0.7307133 ],
     [0.0431029, 0         , 0         , 16.6082821, -5.4293079],
     [0.404103 , 0         , -5.7418613, 0         , 16.5436726]]


*/
