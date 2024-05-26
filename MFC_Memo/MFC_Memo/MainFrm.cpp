
// MainFrm.cpp: CMainFrame 클래스의 구현
//

#include "pch.h"
#include "framework.h"
#include "MFC_Memo.h"

#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMainFrame

IMPLEMENT_DYNCREATE(CMainFrame, CFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWnd)
	ON_WM_CREATE()
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,           // 상태 줄 표시기
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};

// CMainFrame 생성/소멸

CMainFrame::CMainFrame() noexcept
{
	// TODO: 여기에 멤버 초기화 코드를 추가합니다.
}

CMainFrame::~CMainFrame()
{
}

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CFrameWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	if (!m_wndToolBar.CreateEx(this,
		TBSTYLE_FLAT,			// 플랫한 도구 모음 스타일을 지정합니다.
		WS_CHILD				// 자식 윈도우 스타일.
		| NULL					// 도구 모음을 생성할 때 바로 보이도록 설정.
		| CBRS_TOP				// 도구 모음을 창의 상단에 배치.
		| CBRS_GRIPPER			// 도구 모음을 드래그하여 이동할 수 있는 그리퍼를 추가.
		| CBRS_TOOLTIPS			// 도구 모음 버튼에 툴팁을 활성화.
		| CBRS_FLYBY			// 도구 모음 버튼 위에 마우스를 올렸을 때 상태 표시줄에 설명을 표시.
		| CBRS_SIZE_DYNAMIC) || // 도구 모음을 동적으로 크기를 조절할 수 있도록 허용.

		!m_wndToolBar.LoadToolBar(IDR_MAINFRAME)) // IDR_MAINFRAME 메뉴 리소스 로드
	{
		TRACE0("도구 모음을 만들지 못했습니다.\n");
		return -1;      // 만들지 못했습니다.
	}

	if (!m_wndStatusBar.Create(this))
	{
		TRACE0("상태 표시줄을 만들지 못했습니다.\n");
		return -1;      // 만들지 못했습니다.
	}
	m_wndStatusBar.SetIndicators(indicators, sizeof(indicators) / sizeof(UINT));

	// TODO: 도구 모음을 도킹할 수 없게 하려면 이 세 줄을 삭제하십시오.
	m_wndToolBar.EnableDocking(CBRS_ALIGN_ANY);
	EnableDocking(CBRS_ALIGN_ANY);
	DockControlBar(&m_wndToolBar);

	return 0;
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if (!CFrameWnd::PreCreateWindow(cs))
		return FALSE;
	// ************************** 여기 **************************
	CString s("Moon's_MFC_Memo");
	this->SetTitle(s);

	cs.style &= ~FWS_ADDTOTITLE;
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	// ************************** 여기 **************************
	return TRUE;
}

// CMainFrame 진단

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CFrameWnd::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CFrameWnd::Dump(dc);
}
#endif //_DEBUG

// CMainFrame 메시지 처리기