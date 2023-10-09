var data = {
  labels: ["시간당 거북목", "시간당 눈 안 깜빡임"],
  datasets: [
    {
      label: "횟수", //제목
      data: [
        //차트에 들어가는 데이터 값
        0, 0,
      ],
      backgroundColor: [
        //막대 그래프의 배경색
        "rgba(255, 99, 132, 0.2)",
        "rgba(54, 162, 235, 0.2)",
        "rgba(255, 206, 86, 0.2)",
        "rgba(75, 192, 192, 0.2)",
        "rgba(153, 102, 255, 0.2)",
        "rgba(255, 159, 64, 0.2)",
      ],
      borderColor: [
        //막대 그래프를 감싸는 border의 색
        "rgba(255,99,132,1)",
        "rgba(54, 162, 235, 1)",
        "rgba(255, 206, 86, 1)",
        "rgba(75, 192, 192, 1)",
        "rgba(153, 102, 255, 1)",
        "rgba(255, 159, 64, 1)",
      ],
      borderWidth: 1, //border의 넓이
    },
  ],
};
//차트 추가 옵션 값. yAxes 설정 안하면 가운데서 부터 시작하기 때문에 begineAtZero로 설정해야함. canvas크기에 맞춰 생성되기때문에 responsive false로 설정해야함 해당값 없으면 차트가 canvas와 상관없이 사이즈가 커짐
var options = {
  animation: {
    animateScale: true,
  },
  responsive: false,
  scales: {
    yAxes: [
      {
        ticks: {
          beginAtZero: true,
        },
      },
    ],
  },
};

//canvas선택 후 차트 만들어줌 type 변화시키면 차트모양 바꿀 수 있음
var ctx = document.getElementById("upright_chart").getContext("2d");
var myBarChart = new Chart(ctx, {
  type: "bar",
  data: data,
  options: options,
});

//버튼 누르면 ajax요청하고 데이터베이스 값 받아옴
var button = document.getElementById("sendAjax");

button.addEventListener("click", function () {
  sendAjax("/"); //여기 주소가 뭘 의미하는지 잘 모르겠는데 ajax쓰면 주소를 넣더라
});

function sendAjax(url) {
  var oReq = new XMLHttpRequest();

  oReq.open("POST", url);
  oReq.setRequestHeader("Content-Type", "application/json"); // json 형태로 보낸다
  oReq.send();

  oReq.addEventListener("load", function () {
    var result = JSON.parse(oReq.responseText);
    var score = result.score;
    var comp_data = data.datasets[0].data;
    console.log(score);
    for (var i = 0; i < 2; i++) {
      comp_data[i] = score[i];
    }

    data.datasets[0].data = comp_data;
    myBarChart.update();
  });
}
