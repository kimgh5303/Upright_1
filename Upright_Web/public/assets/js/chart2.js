var data2 = {
  labels: [
    "본인 거북목",
    "평균 거북목",
    "본인 눈 안 깜빡임",
    "평균 눈 안 깜빡임",
  ],
  datasets: [
    {
      label: "횟수", //제목
      data: [
        //차트에 들어가는 데이터 값
        0, 0, 0, 0, 0,
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
var options2 = {
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
var ctx2 = document.getElementById("upright_chart2").getContext("2d");
var myBarChart2 = new Chart(ctx2, {
  type: "bar",
  data: data2,
  options: options2,
});

//버튼 누르면 ajax요청하고 데이터베이스 값 받아옴
var button2 = document.getElementById("sendBjax");

button2.addEventListener("click", function () {
  sendBjax("/"); //여기 주소가 뭘 의미하는지 잘 모르겠는데 ajax쓰면 주소를 넣더라
});

function sendBjax(url) {
  var oReq2 = new XMLHttpRequest();

  oReq2.open("POST", url);
  oReq2.setRequestHeader("Content-Type", "application/json"); // json 형태로 보낸다
  oReq2.send();

  oReq2.addEventListener("load", function () {
    var result2 = JSON.parse(oReq2.responseText);
    var score = result2.score;
    var comp_data2 = data2.datasets[0].data;

    for (var i = 2; i < 6; i++) {
      comp_data2[i - 2] = score[i];
    }

    data2.datasets[0].data = comp_data2;
    myBarChart2.update();
  });
}
